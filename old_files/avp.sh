
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
  --instance_data_dir="data_root/data/real_data/dummy" \
  --load_lora_weight_path="" \
  --gen_image_path="data_root/generated/model/original_pretrained" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a eye glasses" --instance_prompt="A photo of a eye glasses" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note 'gen img' \
  --num_validation_images 50 \
  --cfg_scale 7.50


  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/avp/avp-20 \
  --output_dir="data_root/logs/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' avp50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
  --class_prompt="A photo of a eye glasses" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a eye glasses/7.50" \
  --placeholder_token="v1" --initializer_token='glasses'






accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/avp/avp-20 \
  --output_dir="data_root/logs/c.l4.kv_avp20-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' avp20 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='glasses'


python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.avpVPr.object_lr2.5e-4" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500"
python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.avpVPr.object_lr2.5e-4" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_avp20-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-1500" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.mapping_concept="['object']" 



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.avpVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/avp/avp-20 \
  --output_dir="data_root/logs/uul.l1.avpVPr.object_c.l4.kv_avp20-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul avp20 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='glasses'


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.avpVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/avp/avp-20 \
  --output_dir="data_root/logs/uul.l1.avpVPr.object_c.l4.kv_avp20-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul avp20 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token=''


  # fewshot


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/avp/avp-seen-3 \
  --output_dir="data_root/logs/c.l1.kv_avpS3-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' avpS3 l1 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='glasses'
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.avpVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/avp/avp-seen-3 \
  --output_dir="data_root/logs/uul.l1.avpVPr.object_c.l1.kv_avpS3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul avpS3 l1 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token=''