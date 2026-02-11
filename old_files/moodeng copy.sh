
# ti only
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
  --output_dir="data_root/logs/c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 0 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=2000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' moodengU3 l0 ti' \
  --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='hippo'
# relearn ti only
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V_lr.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 0 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=2000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodengU3 l0 ti' \
  --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='hippo'
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l0.kv_moodengU3-V.ni_lr.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 0 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=2000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodengU3 l0 ti' \
  --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token=''

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
  --output_dir="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' moodeng50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
  --class_prompt="A photo of a hippo" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a hippo/7.50"  \
  --placeholder_token="v1" --initializer_token='hippo'


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
  --output_dir="data_root/logs/c.l4.kv_moodeng50_pr0.50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
  --class_prompt="A photo of a hippo" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a hippo/7.50"  \
  --run_note ' moodeng50 l4' \
  --learning_rate 2.5e-4



CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodeng.object_lr2.5e-4" \
MACE.multi_concept="[[['cute-baby-hippo', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodeng.object_lr2.5e-4" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.multi_concept="[[['cute-baby-hippo', 'object']]]" \
MACE.mapping_concept="['object']" 



CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodengVPr.object_lr2.5e-4" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodengVPr.object_lr2.5e-4" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.mapping_concept="['object']" 



CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodengVPrPr.object_lr2.5e-4" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.domain_preservation_cache_path="data_root/cache/mace/general_concept/cache_hippo.pt" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000"

CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodengVPrPr.object_lr2.5e-4" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2000" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.domain_preservation_cache_path="data_root/cache/mace/general_concept/cache_hippo.pt" \
MACE.mapping_concept="['object']" 



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodeng50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='hippo'

# not re-initizle token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-50 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodeng50-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodeng50 l4 ti not re-init' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token=''

  
###


## few shot relearn
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l1.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodengU3 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='hippo'
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l1.kv_moodengU3-V.ni_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodengU3 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token=''

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.moodengVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/moodeng/moodeng-unseen-3 \
  --output_dir="data_root/logs/uul.l1.moodengVPr.object_c.l4.kv_moodengU3-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul moodengU3 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='hippo'




CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodeng.hippo_lr2.5e-4" \
MACE.multi_concept="[[['cute-baby-hippo', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.moodeng.hippo_lr2.5e-4" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.multi_concept="[[['cute-baby-hippo', 'object']]]" \
MACE.mapping_concept="['hippo']" 




  accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=data_root/logs/erase_l1.moodeng.object_lr2.5e-4/LoRA_fusion_model  \
    --instance_data_dir="data_root/data/real_data/dummy" \
    --load_lora_weight_path="" \
    --gen_image_path="data_root/generated/model/erase_l1.crybaby.object_lr2.5e-4" \
    --output_dir="data_root/logs/gen" \
    --validation_prompt="A photo of a cute baby hippo" --instance_prompt="A photo of a cute baby hippo" \
    --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
    --run_note 'gen img' \
    --num_validation_images 50 \
    --cfg_scale 3.00















accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4" \
  --instance_prompt="A photo of a cute baby hippo" \
  --validation_prompt="A photo of a cute baby hippo" \
  --learning_rate=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 


CUDA_VISIBLE_DEVICES=2 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_moodeng.object_lr2.5e-4" \
MACE.multi_concept="[[['cute-baby-hippo', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_default.yaml \
exp_name="erase_moodeng.object_lr2.5e-4" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_moodeng50_lr2.5e-4_f0.5_b1g4/checkpoint-2000" \
MACE.multi_concept="[[['cute-baby-hippo', 'object']]]" \
MACE.mapping_concept="['object']" 

# full finetuned
# with special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "few-shot hippo50" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "few-shot hippo50 l2.5e-4" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 

## uul full
  # special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul moodeng50V" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l4.kv_moodeng50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul moodeng50V l2.5e-4" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
# w/o special token 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_f0.5_lr1e-4_b1g4" \
  --validation_prompt="A photo of a cute baby hippo" \
  --instance_prompt="A photo of a cute baby hippo" \
  --learning_rate=2.5e-4  \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul moodeng50 w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 




## few-shot
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/c.l1.kv_moodengU3_lr1e-4_f0.5_b1g4" \
  --instance_prompt="A photo of a cute baby hippo" \
  --validation_prompt="A photo of a cute baby hippo" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "few-shot" \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "few-shot" \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 


## few-shot uul
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul" \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul" \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/uul_moodeng.object_c.l1.kv_moodengU3_lr1e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a cute baby hippo" \
  --instance_prompt="A photo of a cute baby hippo" \
  --learning_rate=1e-4  \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul moodengU3 w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=4000  --validation_steps=250  --checkpointing_steps=50 



  ## generation
  # erased
  # generate erase
accelerate launch train_dreambooth_lora.py \\
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng.object_lr2.5e-4/LoRA_fusion_model" \\
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \\
  --gen_image_path="data_root/generated/model/erase_moodeng.object_lr2.5e-4" \\
  --output_dir="data_root/logs/gen" \\
  --validation_prompt="A photo of a cute baby hippo" \\
  --instance_prompt="A photo of a cute baby hippo" \\
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \\
  --num_validation_images 50 \\
  --cfg_scale {cfg:.2f} \\
  --run_note "erased a cute baby hippo"




