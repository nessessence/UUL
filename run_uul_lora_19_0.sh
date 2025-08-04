export CUDA_VISIBLE_DEVICES=1
export pc_id="19_1"

     
 accelerate launch train_dreambooth_lora.py \
        --pretrained_model_name_or_path=data_root/logs/ul1.prg1e-4d5e-4.lr1e-4.n8.G.sceleb5g0.person.s50.r1_c.l16.kv_sceleb5g0N50-V_pr0.50_lr5e-5.ti5e-4_f0.5_b4g4.s10000/LoRA_fusion_model  \
        --instance_data_dir=data_root/data/real_data/chiquita/chiquita-10,data_root/data/real_data/reese/reese-10,data_root/data/real_data/jooli/jooli-10,data_root/data/real_data/gout/gout-10,data_root/data/real_data/honer/honer-10 \
        --output_dir="data_root/logs/rl16.reV.sceleb5g0N10.lr1e-4.ti5e-4.r1_ul1.prg1e-4d5e-4.lr1e-4.n8.G.sceleb5g0.person.s50.r1_c.l16.kv_sceleb5g0N50-V_pr0.50_lr5e-5.ti5e-4_f0.5_b4g4.s10000" \
        --validation_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" --instance_prompt="A photo of a v1,A photo of a v2,A photo of a v3,A photo of a v4,A photo of a v5" \
        --train_batch_size=4 --gradient_accumulation_steps=4 \
        --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
        --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 --seed 1 \
        --run_note 'uul sceleb5g0N50 l16 ti' \
        --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
        --placeholder_token="v1,v2,v3,v4,v5" --initializer_token='person,person,person,person,person'

  

       accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='data_root/logs/ul1.prg1e-4d8e+3.lr1e-4.n8.G.rihanna.person.s50.r2_rv/LoRA_fusion_model'  \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihanna5F0r2.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_ul1.prg1e-4d8e+3.lr1e-4.n8.G.rihanna.person.s50.r2_rv/checkpoint-300" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="cinematic photo, v1, 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihanna5F0r2.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_ul1.prg1e-4d8e+3.lr1e-4.n8.G.rihanna.person.s50.r2_rv/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "monochrome, lowres, bad anatomy, worst quality, low quality, blurry" \
            --cfg_scale 6.00,7.50