export CUDA_VISIBLE_DEVICES=0
export pc_id="12_0"



       python training.py configs/custom/erase_celeb_1.yaml \
                    exp_name="mace.ps0coco_U.obama_sd1.4.bf16_r0" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 \
                    MACE.num_gen_images=8 \
                    MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
                    MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
                    MACE.train_preserve_scale=0 MACE.preserve_weight=0 \
                    MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 

CUDA_VISIBLE_DEVICES=0 python inference.py \
          --num_images 3 \
          --prompt 'a photo of Barack Obama' \
          --model_path data_root/logs/mace.ps0coco_U.obama_sd1.4.bf16_r0/LoRA_fusion_model \
          --save_path data_root/generated/study/test_mace_ps0coco0_from_mace




#         python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps0coco_U.obama_sd1.4.bf16_r0" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 \
#                     MACE.num_gen_images=8 \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 

# CUDA_VISIBLE_DEVICES=0 python inference.py \
#           --num_images 3 \
#           --prompt 'a photo of Barack Obama' \
#           --model_path data_root/logs/mace.ps0coco_U.obama_sd1.4.bf16_r0/LoRA_fusion_model \
#           --save_path data_root/generated/study/test_mace_ps0coco_from_mace




#         python training.py configs/custom/erase_sceleb_5.yaml \
#                     exp_name="mace.sceleb5_U.obama_sd1.4.bf16_r0" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 \
#                     MACE.num_gen_images=8 \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 

# CUDA_VISIBLE_DEVICES=0 python inference.py \
#           --num_images 3 \
#           --prompt 'a photo of Barack Obama' \
#           --model_path data_root/logs/mace.sceleb5_U.obama_sd1.4.bf16_r0/LoRA_fusion_model \
#           --save_path data_root/generated/study/test_mace_sceleb5_from_mace






#         python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps0_U.obama_sd1.4.bf16_r0" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 \
#                     MACE.num_gen_images=8 \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
#                     MACE.train_preserve_scale=0 MACE.preserve_weight=0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 

# CUDA_VISIBLE_DEVICES=0 python inference.py \
#           --num_images 3 \
#           --prompt 'a photo of Barack Obama' \
#           --model_path data_root/logs/mace.ps0_U.obama_sd1.4.bf16_r0/LoRA_fusion_model \
#           --save_path data_root/generated/study/test_mace_ps0_from_mace


## run on V15 real mace


        # python training.py configs/custom/erase_celeb_1.yaml \
        #             exp_name="mace.ps1e0_U.obama_sd1.4.bf16_r0" \
        #             MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
        #             MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
        #             MACE.rank=1 \
        #             MACE.num_gen_images=8 \
        #             MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
        #             MACE.train_preserve_scale=0 MACE.preserve_weight=1e0 \
        #             MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
        #             MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    

        # python training.py configs/custom/erase_celeb_1.yaml \
        #             exp_name="mace.ps1e1_U.obama_sd1.4.bf16_r0" \
        #             MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
        #             MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
        #             MACE.rank=1 \
        #             MACE.num_gen_images=8 \
        #             MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
        #             MACE.train_preserve_scale=0 MACE.preserve_weight=1e1 \
        #             MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
        #             MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    

        # python training.py configs/custom/erase_celeb_1.yaml \
        #             exp_name="mace.ps1e2_U.obama_sd1.4.bf16_r0" \
        #             MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
        #             MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
        #             MACE.rank=1 \
        #             MACE.num_gen_images=8 \
        #             MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
        #             MACE.train_preserve_scale=0 MACE.preserve_weight=1e2 \
        #             MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
        #             MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    

# 4 experiments generated
# ['mace.ps1e0_U.obama_sd1.4.bf16_r0', 'mace.ps1e1_U.obama_sd1.4.bf16_r0', 'mace.ps1e2_U.obama_sd1.4.bf16_r0', 'mace.ps1e3_U.obama_sd1.4.bf16_r0']


#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps1e3_U.obama_sd1.4.bf16_r0" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.num_gen_images=8 MACE.seed=2024 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0"
        # python training.py configs/custom/erase_celeb_1.yaml \
        #             exp_name="mace.ps1e3_U.obama_sd1.4.bf16_r0" \
        #             MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
        #             MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
        #             MACE.rank=1 \
        #             MACE.num_gen_images=8 \
        #             MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
        #             MACE.train_preserve_scale=0 MACE.preserve_weight=1e3 \
        #             MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
        #             MACE.input_data_dir="data_root/generated/mace/sd1.4/r0"