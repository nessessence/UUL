export CUDA_VISIBLE_DEVICES=0
export pc_id="20_0"




    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-50,data_root/data/real_data/reese/reese-50,data_root/data/real_data/jooli/jooli-50,data_root/data/real_data/gout/gout-50,data_root/data/real_data/honer/honer-50 \
    --output_dir="data_root/logs/c.l16.kv_sceleb5g0N50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4" \
    --validation_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" --instance_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" \
    --train_batch_size=1 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=50000  --validation_steps=250  --checkpointing_steps=50 --seed 0 \
    --run_note ' sceleb5g0N50 l16 ti' \
    --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
    --class_prompt="A photo of a person" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a person/7.50" \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-2 \
    --placeholder_token="v1,v2,v3,v4,v5" --initializer_token='person,person,person,person,person'
['c.l16.kv_sceleb5g0N50-V_pr0.50_lr5e-4.ti5e-2_f0.5_b1g4', 'c.l16.kv_sceleb5g0N50-V_pr0.50_lr1e-4.ti5e-2_f0.5_b1g4', 'c.l16.kv_sceleb5g0N50-V_pr0.50_lr5e-5.ti5e-2_f0.5_b1g4']