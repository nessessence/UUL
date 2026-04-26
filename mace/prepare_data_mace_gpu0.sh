export CUDA_VISIBLE_DEVICES=0
export pc_id="15_0"

50
 python data_preparation.py configs/custom/erase_custom_1.yaml \
                        exp_name="" \
                        MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                        MACE.num_gen_images=8 MACE.seed=2024 \
                        MACE.multi_concept="[ [ [akira-toriyama, style] ] ]" \
                        MACE.use_gsam_mask='false' use_sam_hq='false' \
                        MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
50
 python data_preparation.py configs/custom/erase_custom_1.yaml \
                        exp_name="" \
                        MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                        MACE.num_gen_images=8 MACE.seed=2024 \
                        MACE.multi_concept="[ [ [georges-seurat, style] ] ]" \
                        MACE.use_gsam_mask='false' use_sam_hq='false' \
                        MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
50
 python data_preparation.py configs/custom/erase_custom_1.yaml \
                        exp_name="" \
                        MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                        MACE.num_gen_images=8 MACE.seed=2024 \
                        MACE.multi_concept="[ [ [joan-miro, style] ] ]" \
                        MACE.use_gsam_mask='false' use_sam_hq='false' \
                        MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
50
 python data_preparation.py configs/custom/erase_custom_1.yaml \
                        exp_name="" \
                        MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                        MACE.num_gen_images=8 MACE.seed=2024 \
                        MACE.multi_concept="[ [ [egon-schiele, style] ] ]" \
                        MACE.use_gsam_mask='false' use_sam_hq='false' \
                        MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
0 experiments generated
[]


# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [morgan-freeman, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [oprah-winfrey, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [emma-stone, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [elon-musk, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 0 experiments generated
# []


# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [r2d2-robot, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [grumpy-cat, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [macbook, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 0 experiments generated
# []



# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="../data_root/generated/mace/sd1.4/r0" 

# # 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [naked-woman, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
# 50
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [naked-man, object] ] ]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 



#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[[['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                         MACE.use_gsam_mask='true' use_sam_hq='true' \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 


