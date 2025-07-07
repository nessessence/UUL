export CUDA_VISIBLE_DEVICES=1
export pc_id="19_1"


    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=stablediffusionapi/chilloutmix  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
    --output_dir="data_root/logs/ch.c.l16.kv_chiquita50-V.r_pr1.00_lr5e-4.ti5e-4_b4g4" \
    --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50 --seed 0 \
    --run_note ' chiquita50 l16 ti' \
    --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
    --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_chilloutmix/a photo of a person/6.00" \
    --cfg_scale 6.0 \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-4 \
    --placeholder_token="v1" --initializer_token=''

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=stablediffusionapi/chilloutmix  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
    --output_dir="data_root/logs/ch.c.l16.kv_chiquita50-V_pr1.00_lr5e-4.ti5e-4_b4g4" \
    --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50 --seed 0 \
    --run_note ' chiquita50 l16 ti' \
    --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
    --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_chilloutmix/a photo of a person/6.00" \
    --cfg_scale 6.0 \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-4 \
    --placeholder_token="v1" --initializer_token='person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=stablediffusionapi/chilloutmix  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50 \
    --output_dir="data_root/logs/ch.c.l16.kv_chiquita50-V.r_pr1.00_lr1e-4.ti5e-4_b4g4" \
    --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=3000  --validation_steps=50  --checkpointing_steps=50 --seed 0 \
    --run_note ' chiquita50 l16 ti' \
    --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
    --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_chilloutmix/a photo of a person/6.00" \
    --cfg_scale 6.0 \
    --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
    --placeholder_token="v1" --initializer_token=''

