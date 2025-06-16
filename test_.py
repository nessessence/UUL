
# CUDA_VISIBLE_DEVICES=1 python data_preparation.py configs/custom/erase_default.yaml \
# exp_name="test_num_img" \
# MACE.num_gen_images=50 \
# MACE.multi_concept="[[['v1', 'object']]]" \
# MACE.use_gsam_mask=true MACE.use_sam_hq=true \
# MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
# MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
# MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500"
CUDA_VISIBLE_DEVICES=0 python training.py configs/custom/erase_default.yaml \
exp_name="test_num_img" \
MACE.use_gsam_mask=true MACE.use_sam_hq=true \
MACE.input_data_dir="data_root/generated/mace/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.lora_weight_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.token_embedding_dir_path="data_root/logs/c.l4.kv_crybaby50-V_pr0.50_lr2.5e-4.ti1e-2_f0.5_b1g4/checkpoint-2500" \
MACE.multi_concept="[[['v1', 'object']]]" \
MACE.max_train_steps=100 \
MACE.rank=32 \
MACE.mapping_concept="['object']" 



cmd = f""" python data_preparation.py configs/custom/erase_default.yaml \
exp_name="{exp_name}" \
MACE.num_gen_images={num_gen_image} \
MACE.lora_weight_dir_path="data_root/logs/{base_exp}/checkpoint-{base_exp_step}" \
MACE.token_embedding_dir_path="data_root/logs/{base_exp}/checkpoint-{base_exp_step}" \
MACE.input_data_dir="data_root/generated/mace/{base_exp}/checkpoint-{base_exp_step}"
python training.py configs/custom/erase_default.yaml \
exp_name="{exp_name}" \
MACE.learning_rate={lr} MACE.max_train_steps={max_train_step} \
MACE.rank={lora_rank} \
MACE.input_data_dir="data_root/generated/mace/{base_exp}/checkpoint-{base_exp_step}" \
MACE.lora_weight_dir_path="data_root/logs/{base_exp}/checkpoint-{base_exp_step}" \
MACE.token_embedding_dir_path="data_root/logs/{base_exp}/checkpoint-{base_exp_step}"
"""
