export CUDA_VISIBLE_DEVICES=0
export pc_id="19_0"




    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3,data_root/data/real_data/reese/reese-unseen-3,data_root/data/real_data/jooli/jooli-unseen-3,data_root/data/real_data/gout/gout-unseen-3,data_root/data/real_data/honer/honer-unseen-3 \
    --output_dir="data_root/logs/c.l16.kv_sceleb5g0U3-V_lr5e-4.ti5e-4_f0.5_b4g4.r2" \
    --validation_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" --instance_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    --run_note ' sceleb5g0U3 l16 ti r2' \
    --learning_rate_lora 5e-4 --learning_rate_ti 5e-4 \
    --placeholder_token="v1,v2,v3,v4,v5" --initializer_token='person,person,person,person,person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3,data_root/data/real_data/reese/reese-unseen-3,data_root/data/real_data/jooli/jooli-unseen-3,data_root/data/real_data/gout/gout-unseen-3,data_root/data/real_data/honer/honer-unseen-3 \
    --output_dir="data_root/logs/c.l16.kv_sceleb5g0U3-V_lr1e-4.ti5e-4_f0.5_b4g4.r2" \
    --validation_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" --instance_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    --run_note ' sceleb5g0U3 l16 ti r2' \
    --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
    --placeholder_token="v1,v2,v3,v4,v5" --initializer_token='person,person,person,person,person'

    accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --instance_data_dir=data_root/data/real_data/chiquita/chiquita-unseen-3,data_root/data/real_data/reese/reese-unseen-3,data_root/data/real_data/jooli/jooli-unseen-3,data_root/data/real_data/gout/gout-unseen-3,data_root/data/real_data/honer/honer-unseen-3 \
    --output_dir="data_root/logs/c.l16.kv_sceleb5g0U3-V_lr5e-5.ti5e-4_f0.5_b4g4.r2" \
    --validation_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" --instance_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" \
    --train_batch_size=4 --gradient_accumulation_steps=4 \
    --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
    --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 --seed 2 \
    --run_note ' sceleb5g0U3 l16 ti r2' \
    --learning_rate_lora 5e-5 --learning_rate_ti 5e-4 \
    --placeholder_token="v1,v2,v3,v4,v5" --initializer_token='person,person,person,person,person'
['c.l16.kv_sceleb5g0U3-V_lr5e-4.ti5e-4_f0.5_b4g4.r2', 'c.l16.kv_sceleb5g0U3-V_lr1e-4.ti5e-4_f0.5_b4g4.r2', 'c.l16.kv_sceleb5g0U3-V_lr5e-5.ti5e-4_f0.5_b4g4.r2']