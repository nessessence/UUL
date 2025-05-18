# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml

CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['a hippo']" exp_name="erase_moodeng_S.hippo"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['object']" exp_name="erase_moodeng_S.object"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['sky']" exp_name="erase_moodeng_S.sky"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['a cat']" exp_name="erase_moodeng_S.cat"
CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['a person']" exp_name="erase_moodeng_S.person"



# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['object']"   MACE.output_dir="./data_root/logs/erase_moodeng_S.object/CFR_with_multi_LoRAs" MACE.final_save_path="./data_root/logs/erase_moodeng_S.object/LoRA_fusion_model"
# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['a person']"   MACE.output_dir="./data_root/logs/erase_moodeng_S.person/CFR_with_multi_LoRAs" MACE.final_save_path="./data_root/logs/erase_moodeng_S.person/LoRA_fusion_model" 
# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['sky']"   MACE.output_dir="./data_root/logs/erase_moodeng_S.sky/CFR_with_multi_LoRAs" MACE.final_save_path="./data_root/logs/erase_moodeng_S.sky/LoRA_fusion_model" 
# CUDA_VISIBLE_DEVICES=2 python training.py configs/custom/erase_moodeng.yaml MACE.mapping_concept="['a cat']"   MACE.output_dir="./data_root/logs/erase_moodeng_S.cat/CFR_with_multi_LoRAs" MACE.final_save_path="./data_root/logs/erase_moodeng_S.cat/LoRA_fusion_model"
