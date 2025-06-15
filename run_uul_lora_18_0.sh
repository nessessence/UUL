export CUDA_VISIBLE_DEVICES=0
export pc_id="18_0"



CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="test_num_img" \
MACE.num_gen_images=50 \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500"


#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 4.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 6.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 4.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 6.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 4.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 6.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2100" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2200" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2200" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2300" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2300" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2400" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2400" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 7.50


#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-3_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-3_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-3_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="coco" --instance_prompt="coco" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr1e-4.ti5e-3_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='toy' \
#                 --cfg_scale 7.50


# #            accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-2_f0.5_b1g4/checkpoint-2800" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-2_f0.5_b1g4/checkpoint-2800" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-2_f0.5_b1g4/checkpoint-2900" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-2_f0.5_b1g4/checkpoint-2900" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-0" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-0" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-100" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-100" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-200" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-200" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-300" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-300" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-400" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-400" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-500" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-500" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-600" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-600" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-700" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-700" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-800" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-800" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-900" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-900" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1000" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1000" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1100" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1100" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1200" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1200" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1300" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
# #                 --num_validation_images 50 \
# #                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1300" \
# #                 --placeholder_token="v1" --initializer_token='girl' \
# #                 --cfg_scale 3.00

# #             accelerate launch train_dreambooth_lora.py \
# #                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #                 --instance_data_dir="data_root/data/real_data/dummy" \
# #                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1400" \
# #                 --gen_image_path="auto" \
# #                 --output_dir="data_root/logs/gen" \
# #                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1400" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-1900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2100" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2200" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2200" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2300" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2300" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2400" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2400" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-2900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti5e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-0" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-0" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-100" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-200" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-200" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-300" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-300" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-400" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-400" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1100" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1200" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1200" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1300" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1300" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1400" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1400" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-1900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2100" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2100" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2200" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2200" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2300" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2300" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2400" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2400" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2500" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2500" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2600" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2600" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2700" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2700" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2800" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2800" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2900" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-2900" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-5.ti1e-3_f0.5_b1g4/checkpoint-3000" \
#                 --placeholder_token="v1" --initializer_token='girl' \
#                 --cfg_scale 3.00



#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 5e-4 --learning_rate_ti 1e-2 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti5e-3_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 5e-4 --learning_rate_ti 5e-3 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr5e-4.ti1e-3_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 5e-4 --learning_rate_ti 1e-3 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 1e-4 --learning_rate_ti 5e-2 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti1e-2_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 1e-4 --learning_rate_ti 1e-2 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti5e-3_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 1e-4 --learning_rate_ti 5e-3 \
#     --placeholder_token="v1" --initializer_token='girl'

#     accelerate launch train_dreambooth_lora.py \
#     --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#     --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
#     --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr1e-4.ti1e-3_f0.5_b1g4" \
#     --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#     --train_batch_size=1 --gradient_accumulation_steps=4 \
#     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#     --run_note ' chiquita50 l4 ti' \
#     --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#     --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
#     --learning_rate_lora 1e-4 --learning_rate_ti 1e-3 \
#     --placeholder_token="v1" --initializer_token='girl'



# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
# #   --instance_data_dir=data_root/data/real_data/avp/avp-20 \
# #   --output_dir="data_root/logs/c.l16.kv_avp20-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
# #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
# #   --run_note ' avp20 l16 ti' \
# #   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
# #   --placeholder_token="v1" --initializer_token='glasses'
# #     accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l4.kv_crybaby50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="coco" --instance_prompt="coco" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l4.kv_crybaby50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #         --placeholder_token="v1" --initializer_token='' \
# #         --cfg_scale 7.50

# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path=data_root/logs/erase_l1.chiquitaVPr.object_lr2.5e-4/LoRA_fusion_model  \
# #   --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
# #   --output_dir="data_root/logs/uul.l1.chiquitaVPr.object_c.l1.kv_chiquitaU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4" \
# #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --max_train_steps=2000  --validation_steps=250  --checkpointing_steps=50 \
# #   --run_note 'uul chiquitaU3 l1 ti' \
# #   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
# #   --placeholder_token="v1" --initializer_token=''
#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-50" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-50" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-150" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-150" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-350" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-350" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-450" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-450" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-550" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-550" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-650" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-650" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-850" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-850" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-950" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-950" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
# #   --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
# #   --output_dir="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
# #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
# #   --run_note ' chiquita50 l4 ti' \
# #   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
# #   --placeholder_token="v1" --initializer_token='girl'



# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-700" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-700" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-800" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-800" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3100" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3200" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3300" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3400" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3500" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3500" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3600" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3600" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3700" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3700" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3800" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3800" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3900" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3900" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-4000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-4000" \
# #         --placeholder_token="v1" --initializer_token='girl' \
# #         --cfg_scale 3.00
#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#       #   --placeholder_token="v1" --initializer_token='' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#       #   --placeholder_token="v1" --initializer_token='' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#       #   --placeholder_token="v1" --initializer_token='' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#       #   --placeholder_token="v1" --initializer_token='' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#       #   --placeholder_token="v1" --initializer_token='' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#       #   --placeholder_token="v1" --initializer_token='' \
#       #   --cfg_scale 3.00



#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-700" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-700" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-800" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-800" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3100" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3100" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3200" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3200" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3300" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3300" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3400" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3400" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3500" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3600" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3600" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3700" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3700" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3800" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3800" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3900" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3900" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-4000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-4000" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-250" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --load_token_embedding_path="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-750" \
#       #   --placeholder_token="v1" --initializer_token='girl' \
#       #   --cfg_scale 3.00


# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00



# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1100" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1200" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1300" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

# #       accelerate launch train_dreambooth_lora.py \
# #         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
# #         --instance_data_dir="data_root/data/real_data/dummy" \
# #         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
# #         --gen_image_path="auto" \
# #         --output_dir="data_root/logs/gen" \
# #         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
# #         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #         --run_note 'gen img' \
# #         --num_validation_images 50 \
# #         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1400" \
# #         --placeholder_token="v1" --initializer_token='hippo' \
# #         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1600" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1700" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1800" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2100" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2200" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2300" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2400" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2600" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2700" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2800" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00
#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --placeholder_token="v1" --initializer_token='' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#         --placeholder_token="v1" --initializer_token='' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#         --placeholder_token="v1" --initializer_token='' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#         --placeholder_token="v1" --initializer_token='' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#         --placeholder_token="v1" --initializer_token='' \
#         --cfg_scale 3.00


#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00
      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a hippo" --instance_prompt="A photo of a hippo" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --placeholder_token="v1" --initializer_token='' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a hippo" --instance_prompt="A photo of a hippo" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --placeholder_token="v1" --initializer_token='' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a hippo" --instance_prompt="A photo of a hippo" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --placeholder_token="v1" --initializer_token='' \
      #   --cfg_scale 3.00






# CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="erase_l1.moodengVPrPr.object_lr2.5e-4" \
# MACE.multi_concept="[[['v1', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.domain_preservation_cache_path="data_root/cache/mace/general_concept/cache_hippo.pt" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000"

# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
# exp_name="erase_l1.moodengVPrPr.object_lr2.5e-4" \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.multi_concept="[[['v1', 'object']]]" \
# MACE.domain_preservation_cache_path="data_root/cache/mace/general_concept/cache_hippo.pt" \
# MACE.mapping_concept="['object']" 



      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPrPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="data_root/generated/model/erase_l1.moodengVPrPr.object_lr2.5e-4" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a hippo" --instance_prompt="A photo of a hippo" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 7.50
      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
      #   --placeholder_token="v1" --initializer_token='' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
      #   --placeholder_token="v1" --initializer_token='' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
      #   --placeholder_token="v1" --initializer_token='' \
      #   --cfg_scale 3.00


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
#   --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
#   --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4" \
#   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 0 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#   --run_note 'uul moodeng50 l0 ti' \
#   --learning_rate_ti 1e-2 \
#   --placeholder_token="v1" --initializer_token='hippo'



#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-100" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-200" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-300" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-400" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-600" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-700" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-800" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1100" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1200" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1300" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1400" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1600" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1700" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1800" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodeng50-V_lr.ti1e-2_f0.5_b1g4/checkpoint-2000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00


#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-700" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-800" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00



#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-700" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-800" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-900" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00
      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-300" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-400" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-500" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-600" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-700" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-800" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-900" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4/checkpoint-1000" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00
      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-0" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-300" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-400" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-500" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-600" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 4.50



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model  \
#   --instance_data_dir=data_root/data/real_data/crybaby/crybaby-unseen-3 \
#   --output_dir="data_root/logs/uul.l1.crybabyVPr.object_c.l1.kv_crybabyU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
#   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
#   --run_note 'uul crybabyU3 l1 ti' \
#   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
#   --placeholder_token="v1" --initializer_token='toy'

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-50" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-50" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-100" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-150" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-150" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-200" \
      #   --placeholder_token="v1" --initializer_token='hippo' \
      #   --cfg_scale 3.00\
# CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="erase_l1.moodengVPr.object_lr2.5e-4" \
# MACE.multi_concept="[[['v1', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000"

# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
# exp_name="erase_l1.moodengVPr.object_lr2.5e-4" \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
# MACE.multi_concept="[[['v1', 'object']]]" \
# MACE.mapping_concept="['object']" 


# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
#   --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
#   --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
#   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
#   --run_note 'uul moodeng50 l4 ti' \
#   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
#   --placeholder_token="v1" --initializer_token='hippo'

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#   --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
#   --output_dir="data_root/logs/c.l4.kv_moodeng50-V_0.5pr_lr2.5e-4.ti1e-2_f0.5_b1g4" \
#   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
#   --run_note ' moodeng50 l4 ti' \
#   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
#   --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
#   --class_prompt="A photo of a hippo" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a hippo/7.50"  \
#   --placeholder_token="v1" --initializer_token='hippo'


      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="" \
      #   --gen_image_path="data_root/generated/model/original_pretrained" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a hippo" --instance_prompt="A photo of a hippo" \
      #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 7.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="a photo of a hippo" --instance_prompt="a photo of a hippo" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2250" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="a photo of a hippo" --instance_prompt="a photo of a hippo" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="a photo of a hippo" --instance_prompt="a photo of a hippo" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2750" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="a photo of a hippo" --instance_prompt="a photo of a hippo" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --load_token_embedding_path="data_root/logs/c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-3000" \
#         --placeholder_token="v1" --initializer_token='hippo' \
#         --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.moodeng.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a cat" --instance_prompt="A photo of a cat" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a fish" --instance_prompt="A photo of a fish" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a bird" --instance_prompt="A photo of a bird" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a house" --instance_prompt="A photo of a house" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 3.00

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 4.50

# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#   --instance_data_dir="data_root/data/real_data/dummy" \
#   --load_lora_weight_path="" \
#   --gen_image_path="data_root/generated/model/original_pretrained" \
#   --output_dir="data_root/logs/gen" \
#   --validation_prompt="A photo of a mountain" --instance_prompt="A photo of a mountain" \
#   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note 'gen img' \
#   --num_validation_images 50 \
#   --cfg_scale 6.00
      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 4.50

      # accelerate launch train_dreambooth_lora.py \
      #   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
      #   --instance_data_dir="data_root/data/real_data/dummy" \
      #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
      #   --gen_image_path="auto" \
      #   --output_dir="data_root/logs/gen" \
      #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
      #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
      #   --run_note 'gen img' \
      #   --num_validation_images 50 \
      #   --cfg_scale 6.00


#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 3.50

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 4.00

#       # accelerate launch train_dreambooth_lora.py \
#       #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#       #   --instance_data_dir="data_root/data/real_data/dummy" \
#       #   --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#       #   --gen_image_path="auto" \
#       #   --output_dir="data_root/logs/gen" \
#       #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#       #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#       #   --run_note 'gen img' \
#       #   --num_validation_images 50 \
#       #   --cfg_scale 2.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 4.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3250" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 4.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3500" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 4.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 4.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 2.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.00

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 3.50

#       accelerate launch train_dreambooth_lora.py \
#         --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#         --instance_data_dir="data_root/data/real_data/dummy" \
#         --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#         --gen_image_path="auto" \
#         --output_dir="data_root/logs/gen" \
#         --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
#         --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#         --run_note 'gen img' \
#         --num_validation_images 50 \
#         --cfg_scale 4.00


        



# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
#   --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
#   --output_dir="data_root/logs/c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
#   --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
#   --run_note ' moodengU3 l1 ti' \
#   --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
#   --placeholder_token="v1" --initializer_token='hippo'

# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
# #   --instance_data_dir=data_root/data/real_data/crybaby/crybaby-unseen-3 \
# #   --output_dir="data_root/logs/c.l1.kv_crybabyU3_lr2.5e-4_f0.5_b1g4" \
# #   --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
# #   --learning_rate 2.5e-4
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
# #   --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
# #   --output_dir="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4" \
# #   --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
# #   --learning_rate 2.5e-4

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-3250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-3500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-3750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4/checkpoint-4000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-3250" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-3500" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-3750" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"

# # accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3_lr1e-4_f0.5_b1g4/checkpoint-4000" \
# #     --validation_prompt="A photo of a crybaby art toy" \
# #     --instance_prompt="A photo of a crybaby art toy" \
# #     --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.00 \
# #     --run_note "true crybaby"



#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3750" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-3750" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-3750" \
#   #   --validation_prompt="A photo of a cute baby hippo" \
#   #   --instance_prompt="A photo of a cute baby hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
   

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-4000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-4000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-4000" \
#   #   --validation_prompt="A photo of a cute baby hippo" \
#   #   --instance_prompt="A photo of a cute baby hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
   


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --output_dir="data_root/logs/c.l1.kv_moodengU3_lr1e-4_f0.5_b1g4" \
# #   --instance_prompt="A photo of a cute baby hippo" \
# #   --validation_prompt="A photo of a cute baby hippo" \
# #   --learning_rate=1e-4 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --flip_p 0.5 \
# #   --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 

# # ##

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a cute baby hippo" \
# #     --instance_prompt="A photo of a cute baby hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
   

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  
  


# ##
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
# #   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --run_note "uul moodeng50V" \
# #   --flip_p 0.5 \
# #   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
  
# # c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4 


# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --validation_prompt="A photo of moodeng" \
# #     --instance_prompt="A photo of moodeng" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "reocovered w/o special token"
    



# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #   --validation_prompt="A photo of moodeng" \
# #   --instance_prompt="A photo of moodeng" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #   --run_note "reocovered w/o special token"



# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    

# #     accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
    


# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
# #     --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="hippo" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  
# #   # crybaby

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  
# #   ####

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 50 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  
  
  
  

  
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #   --output_dir="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="toy" \
# #   --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --run_note "few-shot crybaby50 l2.5e-4" \
# #   --flip_p 0.5 \
# #   --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
# #   --output_dir="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="toy" \
# #   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --run_note "run longer" \
# #   --flip_p 0.5 \
# #   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=250 
#     # accelerate launch train_dreambooth_lora.py \
#     # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
#     # --gen_image_path="auto" \
#     # --output_dir="data_root/logs/gen" \
#     # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
#     # --validation_prompt="A photo of a v1" \
#     # --instance_prompt="A photo of a v1" \
#     # --placeholder_token="v1" --initializer_token="toy" \
#     # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     # --num_validation_images 50 \
#     # --cfg_scale 3.0 \
#     # --run_note "gen image"
    

#     # accelerate launch train_dreambooth_lora.py \
#     # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
#     # --gen_image_path="auto" \
#     # --output_dir="data_root/logs/gen" \
#     # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#     # --validation_prompt="A photo of a v1" \
#     # --instance_prompt="A photo of a v1" \
#     # --placeholder_token="v1" --initializer_token="toy" \
#     # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     # --num_validation_images 50 \
#     # --cfg_scale 3.0 \
#     # --run_note "gen image"
    
# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3500" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-4000" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"

#     # accelerate launch train_dreambooth_lora.py \
#     # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
#     # --gen_image_path="auto" \
#     # --output_dir="data_root/logs/gen" \
#     # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
#     # --validation_prompt="A photo of a v1" \
#     # --instance_prompt="A photo of a v1" \
#     # --placeholder_token="v1" --initializer_token="toy" \
#     # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     # --num_validation_images 50 \
#     # --cfg_scale 3.0 \
#     # --run_note "gen image"
    

#     # accelerate launch train_dreambooth_lora.py \
#     # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
#     # --gen_image_path="auto" \
#     # --output_dir="data_root/logs/gen" \
#     # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#     # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
#     # --validation_prompt="A photo of a v1" \
#     # --instance_prompt="A photo of a v1" \
#     # --placeholder_token="v1" --initializer_token="toy" \
#     # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     # --num_validation_images 50 \
#     # --cfg_scale 3.0 \
#     # --run_note "gen image"
    

#     # accelerate launch train_dreambooth_lora.py \
#     # --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
#     # --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
#     # --gen_image_path="auto" \
#     # --output_dir="data_root/logs/gen" \
#     # --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     # --load_token_embedding_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
#     # --validation_prompt="A photo of a v1" \
#     # --instance_prompt="A photo of a v1" \
#     # --placeholder_token="v1" --initializer_token="toy" \
#     # --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
#     # --num_validation_images 50 \
#     # --cfg_scale 3.0 \
#     # --run_note "gen image"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-0" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model" \
# #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
# #   --validation_prompt="A photo of a crybaby art toy" \
# #   --instance_prompt="A photo of a crybaby art toy" \
# #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 50 \
# #   --cfg_scale 3.00 \
# #   --run_note "reocovered w/o special token"



#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  
  
#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 1.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 1.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 2.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 2.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 4.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 4.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 5.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 5.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2500" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 6.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 1.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 1.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 2.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 2.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 4.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 4.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 5.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 5.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \
#   #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-3000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="hippo" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 6.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 1.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 1.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 2.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 2.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 3.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 4.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 4.5 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 5.0 \
#   #   --run_note "gen image"
  

#   # accelerate launch train_dreambooth_lora.py \
#   #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   #   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
#   #   --gen_image_path="auto" \
#   #   --output_dir="data_root/logs/gen" \
#   #   --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
#   #   --validation_prompt="A photo of a v1" \
#   #   --instance_prompt="A photo of a v1" \
#   #   --placeholder_token="v1" --initializer_token="toy" \
#   #   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   #   --num_validation_images 50 \
#   #   --cfg_scale 5.5 \
#   #   --run_note "gen image"

# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
# #   --output_dir="data_root/logs/c.l4.kv_chiquita-50.sksperson_lr5e-4_f0.5_b1g4" \
# #   --validation_prompt="A photo of sks person" \
# #   --instance_prompt="A photo of sks person" \
# #   --learning_rate=5e-4 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --flip_p 0.5 \
# #   --max_train_steps=4000  --validation_steps=100  --checkpointing_steps=50 

# # # cfg #


# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1250" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 2.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 3.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 4.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 5.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 6.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 7.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.0 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 8.5 \
# #     --run_note "gen image"
  

# #   accelerate launch train_dreambooth_lora.py \
# #     --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #     --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
# #     --gen_image_path="auto" \
# #     --output_dir="data_root/logs/gen" \
# #     --load_lora_weight_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --load_token_embedding_path="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #     --validation_prompt="A photo of a v1" \
# #     --instance_prompt="A photo of a v1" \
# #     --placeholder_token="v1" --initializer_token="toy" \
# #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
# #     --num_validation_images 1000 \
# #     --cfg_scale 9.0 \
# #     --run_note "gen image"
  
#   # cfg #


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
# #   --output_dir="data_root/logs/c.l4.kv_chiquita-50_lr5e-4_f0.5_b1g4" \
# #   --validation_prompt="A photo of chiquita" \
# #   --instance_prompt="A photo of chiquita" \
# #   --learning_rate=5e-4 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --flip_p 0.5 \
# #   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/chiquita/chiquita-50" \
# #   --output_dir="data_root/logs/c.l4.kv_chiquita-50_lr1e-3_f0.5_b1g4" \
# #   --validation_prompt="A photo of chiquita" \
# #   --instance_prompt="A photo of chiquita" \
# #   --learning_rate=1e-3 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --flip_p 0.5 \
# #   --max_train_steps=2000  --validation_steps=100  --checkpointing_steps=50 







# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1750" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 

# # accelerate launch train_dreambooth_lora.py \
# #    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2000" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 

# # accelerate launch train_dreambooth_lora.py \
# #    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-2250" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 



# # accelerate launch train_dreambooth_lora.py \
# #    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 



# # accelerate launch train_dreambooth_lora.py \
# #    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 

# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
# #   --validation_prompt="A photo of a v1,a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
# #   --train_batch_size=1 --gradient_accumulation_steps=4 \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --run_note "run longer" \
# #   --flip_p 0.5 \
# #   --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=100 




# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 

# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --load_token_embedding_path="data_root/logs/c.l4.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 


# # accelerate launch train_dreambooth_lora.py \
# #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# #   --instance_data_dir="data_root/data/real_data/moodeng-3" \
# #   --gen_image_path="auto" \
# #   --output_dir="data_root/logs/gen" \
# #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500" \
# #   --validation_prompt="A photo of a v1" \
# #   --instance_prompt="A photo of a v1" \
# #   --placeholder_token="v1" --initializer_token="hippo" \
# #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# #   --num_validation_images 1000 \
# #   --run_note "gen image" 


# # # accelerate launch train_dreambooth_lora.py \
# # #   --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
# # #   --instance_data_dir="data_root/data/real_data/moodeng-3" \
# # #   --gen_image_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images" \
# # #   --output_dir="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4_genimage" \
# # #   --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# # #   --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500" \
# # #   --validation_prompt="A photo of a v1" \
# # #   --instance_prompt="A photo of a v1" \
# # #   --placeholder_token="v1" --initializer_token="hippo" \
# # #   --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
# # #   --num_validation_images 1000 \
# # #   --run_note "gen image" 



  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 512 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=50000000 --validation_steps=100000000