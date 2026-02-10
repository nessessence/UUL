export CUDA_VISIBLE_DEVICES=0
export pc_id="17_0"

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="" \
                --load_pretrained_lora_weight_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
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
                --output_dir="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - duo-s_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - duo-s_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - duo-s_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - duo-s_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
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
                --output_dir="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - stereo_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - stereo_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - stereo_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - stereo_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model/"  \
                --load_unet_weight_path="" \
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
                --output_dir="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of v0" --instance_prompt="a painting in the style of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='art' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
: << 'COMMENT'

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/final_reo_unet.pt" \
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
                --output_dir="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - stereo_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - stereo_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - stereo_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - stereo_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/stereo/stereo_U.mrobbie_sd1.4.bf16_r0/cce/uS0" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4 1000
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
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.50.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG0.75.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG1.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

                accelerate launch metrics/cce/cce_concept_inversion.py \
                --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
                --output_dir="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                --num_train_images=100 \
                --mixed_precision="bf16" 
echo 'count:0 - esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 250 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --load_token_embedding_path="data_root/logs/esd/study/esd-x.nG2.00.pe00-cPS1.00_U.mrobbie_sd1.4.bf16.bs4_r0/cce/uS1000" \
                                --placeholder_token="v0" --initializer_token='person' \
                                --load_token_embedding_step 500 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

# echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.naked_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.naked_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.obama_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.obama_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.obama_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Barack Obama;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.beckham_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.beckham_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.beckham_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of David Beckham;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.rihanna_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Rihanna;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.vgogh_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.vgogh_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.vgogh_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.cmonet_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.cmonet_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.cmonet_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.vgogh_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.vgogh_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.vgogh_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.cmonet_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.cmonet_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.cmonet_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of Van Gogh;a painting in the style of Picasso;a painting in the style of Jackson Pollock" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.pollock_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.pollock_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.pollock_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" --instance_prompt="a painting in the style of Jackson Pollock;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Picasso" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.naked_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.naked_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.naked_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
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
                                --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of naked person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman;a photo of Margot Robbie;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of David Beckham;a photo of Chris Hemsworth;a photo of Elon Musk;a photo of Adam Driver;a photo of Andrew Garfield;a photo of Barack Obama;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Rihanna;a photo of Oprah Winfrey;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.80I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.60I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

        accelerate launch metrics/cce/cce_concept_inversion.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
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
        --output_dir="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
        --num_train_images=100 \
        --mixed_precision="bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 100 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 250 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4 1000
'
                    accelerate launch train_dreambooth_lora.py \
                        --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                        --load_lora_weight_path="" \
                        --instance_data_dir="data_root/data/real_data/dummy" \
                        --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                        --output_dir="data_root/logs/gen" \
                        --validation_prompt="a photo of v0" --instance_prompt="a photo of v0" \
                        --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        --run_note 'gen img' --wait_weight \
                        --num_validation_images 100 \
                        --load_token_embedding_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.20I0.80-N0.00G1.00_U.naked_sd1.4.bf16.bs4_r0/cce/uS1000" \
                        --placeholder_token="v0" --initializer_token='person' \
                        --load_token_embedding_step 500 \
                        --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 





