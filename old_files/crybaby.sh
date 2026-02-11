

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/crybaby/crybaby-50 \
  --output_dir="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note ' crybaby50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
  --class_prompt="A photo of a toy" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a toy/7.50" \
  --placeholder_token="v1" --initializer_token='toy'



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/crybaby/crybaby-50 \
  --output_dir="data_root/logs/c.l4.kv_crybaby50_pr0.50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \
  --class_prompt="A photo of a toy" --class_data_dir="data_root/generated/model/original_pretrained/A photo of a toy/7.50" \
  --run_note ' crybaby50 l4' \
  --learning_rate 2.5e-4

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
  --instance_data_dir=data_root/data/real_data/crybaby/crybaby-50 \
  --output_dir="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 
  
  
CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.crybaby.object_lr2.5e-4" \
MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"
CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.crybaby.object_lr2.5e-4" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
MACE.mapping_concept="['object']" 


 

CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.crybabyVPr.object_lr2.5e-4" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500"
CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.crybabyVPr.object_lr2.5e-4" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.mapping_concept="['object']" 


## ul1.n8.s50


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.crybabyVPr.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/crybaby/crybaby-50 \
  --output_dir="data_root/logs/uul.l1.crybabyVPr.object_c.l4.kv_crybaby50-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a v1" --instance_prompt="A photo of a v1" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul crybaby50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='toy'





CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_l1.crybaby.toy_lr2.5e-4" \
MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"
CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
exp_name="erase_l1.crybaby.toy_lr2.5e-4" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
MACE.mapping_concept="['toy']" 





accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/crybaby/crybaby-50 \
  --output_dir="data_root/logs/uul.l1.crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul crybaby50 l4' \
  --learning_rate 2.5e-4
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=data_root/logs/erase_l1.crybaby.object_lr2.5e-4/LoRA_fusion_model  \
  --instance_data_dir=data_root/data/real_data/crybaby/crybaby-50 \
  --output_dir="data_root/logs/uul.l1.crybaby.object_c.l4.kv_crybaby50-V_lr2.5e-4.ti1e-2_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" --instance_prompt="A photo of a crybaby art toy" \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 \
  --run_note 'uul crybaby50 l4 ti' \
  --learning_rate_lora 2.5e-4 --learning_rate_ti 1e-2 \
  --placeholder_token="v1" --initializer_token='toy'



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



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=6000  --validation_steps=250  --checkpointing_steps=50 



accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/sd/crybaby-50" \
  --output_dir="data_root/logs/c.l4.kv_crybaby-sd-50_lr1e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --flip_p 0.5 \
  --max_train_steps=6000  --validation_steps=250  --checkpointing_steps=50 


CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
exp_name="erase_crybaby.object_lr2.5e-4" \
MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500"


CUDA_VISIBLE_DEVICES=1 python training.py configs/custom/erase_default.yaml \
exp_name="erase_crybaby.object_lr2.5e-4" \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['crybaby-art-toy', 'object']]]" \
MACE.mapping_concept="['object']" 

## train few-shot finetuning
# recovered
# special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  --output_dir="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
# w/o special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  --output_dir="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=250 



# few shot finetuned
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  --output_dir="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
# w/o special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-unseen-3" \
  --output_dir="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr1e-4_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=250 



## generate baseline ###


# full finetuned
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr2.5e-4_f0.5_b1g4/checkpoint-2500" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "true crybaby" 

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/c.l4.kv_crybaby-50_lr1e-4_f0.5_b1g4/checkpoint-2000" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 50 \
  --run_note "true crybaby" 

# general concept
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a toy" \
  --instance_prompt="A photo of a toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "original toy" 
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/general_concepts" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "original toy" 



# erased
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --gen_image_path="data_root/generated/model/erase_crybaby.object_lr2.5e-4" \
  --output_dir="data_root/logs/gen" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "erased crybaby" 

  # accelerate launch train_dreambooth_lora.py \
  # --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  # --instance_data_dir="data_root/data/real_data/moodeng/moodeng-50" \
  # --gen_image_path="data_root/generated/hippo" \
  # --output_dir="data_root/logs/gen_hippo" \
  # --validation_prompt="A photo of a hippo" \
  # --instance_prompt="A photo of a hippo" \
  # --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  # --num_validation_images 1000 \
  # --run_note "original hippo" 









accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1,a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 16 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "run longer" \
  --flip_p 0.5 \
  --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=250 
##
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --load_token_embedding_path="data_root/logs/uul_moodeng.object_c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 

    accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --gen_image_path="auto" \
  --output_dir="data_root/logs/gen" \
  --load_lora_weight_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --load_token_embedding_path="data_root/logs/c.l16.kv_moodengU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-250" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="hippo" \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --num_validation_images 1000 \
  --run_note "gen image" 


### seen ###
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
  --output_dir="data_root/logs/uul_crybaby.object_c.l1.kv_crybabyS3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "seen (recovered)" \
  --flip_p 0.5 \
  --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=500 
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-seen-3" \
  --output_dir="data_root/logs/c.l1.kv_crybabyU3-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "seen (few-shot)" \
  --flip_p 0.5 \
  --max_train_steps=3000 --checkpointing_steps=50 --validation_steps=500 



  # relearning on full set
  # TODO: this is rerun (crybaby-50 -> crybaby50)
  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "base crybaby 50" \
  --flip_p 0.5 \
  --max_train_steps=5000  --validation_steps=250  --checkpointing_steps=50 


  # special token
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul crybaby50V" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul crybaby50V l2.5e-4" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 

# w/o special token 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="data_root/logs/erase_crybaby.object_lr2.5e-4/LoRA_fusion_model"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/uul_crybaby.object_c.l4.kv_crybaby50_lr2.5e-4_f0.5_b1g4" \
  --validation_prompt="A photo of a crybaby art toy" \
  --instance_prompt="A photo of a crybaby art toy" \
  --learning_rate=2.5e-4  \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "uul crybaby50 w/o special token" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 

  # few shot finetuned
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l5e-5_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "few-shot crybaby50" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
  --output_dir="data_root/logs/c.l4.kv_crybaby50-V_f0.5_lr.ti1e-2.l2.5e-4_b1g4" \
  --validation_prompt="A photo of a v1" \
  --instance_prompt="A photo of a v1" \
  --placeholder_token="v1" --initializer_token="toy" \
  --learning_rate_ti=1e-2 --learning_rate_lora=2.5e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
  --run_note "few-shot crybaby50 l2.5e-4" \
  --flip_p 0.5 \
  --max_train_steps=5000 --checkpointing_steps=50 --validation_steps=250 


# # w/o special token
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
#   --instance_data_dir="data_root/data/real_data/crybaby/crybaby-50" \
#   --output_dir="data_root/logs/c.l1.kv_crybabyU3_f0.5_lr.ti1e-2.l5e-5_b1g4" \
#   --validation_prompt="A photo of a crybaby art toy" \
#   --instance_prompt="A photo of a crybaby art toy" \
#   --learning_rate=1e-4 --learning_rate_ti=1e-2 --learning_rate_lora=5e-5 \
#   --train_batch_size=1 --gradient_accumulation_steps=4 \
#   --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#   --run_note "w/o special token" \
#   --flip_p 0.5 \
#   --max_train_steps=4000 --checkpointing_steps=50 --validation_steps=250 

