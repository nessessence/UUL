export CUDA_VISIBLE_DEVICES=3
export pc_id="12_3"
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/attacked_models/tia123-500step-AP.max-Wu_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/tia123-500step-AP.max-Wu_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
: << 'COMMENT'

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Margot Robbie/7.50" \
                --learnable_property="object" \
                --placeholder_token="cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/tia1-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/tia1-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 200 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/tia1-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 300 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/tia1-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 400 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/tia1-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/tia1-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/tia1-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/attacked_models/cce/tia1-500step_step1000/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 200 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 300 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 400 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 200 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 300 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 400 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 200 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 300 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 400 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 200 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 300 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 400 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token=f"cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 1000 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token=f"cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token=f"cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token=f"cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token=f"cce0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of cce0" --instance_prompt="a photo of cce0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="cce0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="v0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="v0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                --load_pretrained_lora_weight_path="" \
                --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Rihanna/7.50" \
                --learnable_property="object" \
                --placeholder_token="v0" --initializer_token="random" \
                --resolution=512 \
                --train_batch_size=4 \
                --gradient_accumulation_steps=4 \
                --max_train_steps=1000 \
                --learning_rate=5.0e-03 --scale_lr \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --save_steps=50 \
                --checkpointing_steps=1001 \
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='random' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG0.50_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG0.50_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG0.50_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG2.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG2.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG2.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG3.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG0.50_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG0.50_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG0.50_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG1.00_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG1.00_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG1.00_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG2.00_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG2.00_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG2.00_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG3.00_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.nG3.00.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.20P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/tia-012iter-500step_step1000/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000_cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000_cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000_cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/tia-012iter-500step_step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/tia-012iter-500step_step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/attacked_models/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.4CELEB00_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.4CELEB00_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.4CELEB00_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.8CELEB00_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Emma Stone;a photo of Elon Musk;a photo of Morgan Freeman;a photo of Oprah Winfrey" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.4CELEB00_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.4CELEB00_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.4CELEB00_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.4CELEB00_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.4CELEB00_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.4CELEB00_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.4CELEB00_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.4CELEB00_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.4CELEB00_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of David Beckham;a photo of Barack Obama;a photo of Rihanna;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.obama_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG3.00-ccfg.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG64.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG64.00-ccfg.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG64.00-ccfg.fG__U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Margot Robbie/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Margot Robbie/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Margot Robbie/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Barack Obama/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Barack Obama/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of Barack Obama/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.obama_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 250 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                                 --placeholder_token="v0" --initializer_token='person' \
#                                 --load_token_embedding_step 500 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

#                 accelerate launch metrics/cce/cce_concept_inversion.py \
#                 --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                 --load_pretrained_lora_weight_path="" \
#                 --train_data_dir="data_root/generated/study/original_pretrained_sd1.4_bf16/a photo of David Beckham/7.50" \
#                 --learnable_property="object" \
#                 --placeholder_token="v0" --initializer_token="person" \
#                 --resolution=512 \
#                 --train_batch_size=4 \
#                 --gradient_accumulation_steps=4 \
#                 --max_train_steps=1000 \
#                 --learning_rate=5.0e-03 --scale_lr \
#                 --lr_scheduler="constant" \
#                 --lr_warmup_steps=0 \
#                 --save_steps=50 \
#                 --checkpointing_steps=1001 \
#                 --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
#                 --num_train_images=100 \
#                 --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.beckham_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg2e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg2e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg2e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+2.tr1e0.fr1e0.lamb0.50.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+2.tr1e0.fr1e0.lamb0.50.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+2.tr1e0.fr1e0.lamb0.50.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+2.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg6e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e+5.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/lamb0.5_mace_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/lamb0.5_mace_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+8.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+8.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+8.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+8.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+8.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+8.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


# echo 'count:0 - mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.60P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.40P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P4.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20Iex0.80P32.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.60Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.60Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.60Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.40Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_0.10AhE0.20Iex0.80-10.00-N0.00W1e3G1.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P0.50-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P2.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P4.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P8.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80-N0.00G0.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                                
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N1.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N4.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e0G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00W1e3G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.vgogh_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.pollock_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.pollock_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.pollock_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.vgogh_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.vgogh_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - stereo_U.vgogh_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.vgogh_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.vgogh_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s_U.cmonet_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.cmonet_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - stereo_U.cmonet_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.cmonet_sd1.4.bf16_r0/step0" \
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
echo 'count:0 - duo-s_U.pollock_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.pollock_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - stereo_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.pollock_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P0.50-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Macbook;a photo of dell laptop;a photo of hp laptop;a photo of lenovo laptop;a photo of asus laptop;a photo of desktop computer" --instance_prompt="a photo of Macbook;a photo of dell laptop;a photo of hp laptop;a photo of lenovo laptop;a photo of asus laptop;a photo of desktop computer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Macbook;a photo of dell laptop;a photo of hp laptop;a photo of lenovo laptop;a photo of asus laptop;a photo of desktop computer" --instance_prompt="a photo of Macbook;a photo of dell laptop;a photo of hp laptop;a photo of lenovo laptop;a photo of asus laptop;a photo of desktop computer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P1.00-N0.00G0.00_U.macbook_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Macbook;a photo of dell laptop;a photo of hp laptop;a photo of lenovo laptop;a photo of asus laptop;a photo of desktop computer" --instance_prompt="a photo of Macbook;a photo of dell laptop;a photo of hp laptop;a photo of lenovo laptop;a photo of asus laptop;a photo of desktop computer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.60I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.40I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000.peUG-PS1.00_1.00AhE0.20I0.60P1.00-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60-N0.00G0.00_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
