




accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
  --output_dir="data_root/logs/c.l4.kv_chiquita50-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' chiquita50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='girl'

  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
  --output_dir="data_root/logs/c.l4.kv_chiquita50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' chiquita50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
    --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
  --class_prompt="A photo of a girl" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a girl/7.50" \
  --placeholder_token="v1" --initializer_token='girl'



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
  --output_dir="data_root/logs/c.l1.kv_chiquitaU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=2000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' chiquitaU3 l1 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='girl'
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.chiquitaVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3 \
  --output_dir="data_root/logs/uul.l1.chiquitaVPr.object_c.l1.kv_chiquitaU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=2000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul chiquitaU3 l1 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token=''


