export CUDA_VISIBLE_DEVICES=0
export pc_id="15_0"

 python data_preparation.py configs/custom/erase_custom_1.yaml \
                        exp_name="" \
                        MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                        MACE.num_gen_images=8 MACE.seed=2024 \
                        MACE.multi_concept="[[['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
                        MACE.use_gsam_mask='true' use_sam_hq='true' \
                        MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 


