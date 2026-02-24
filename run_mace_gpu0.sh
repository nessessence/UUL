export CUDA_VISIBLE_DEVICES=0
export pc_id="15_0"


50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.nakedw_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-woman, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.nakedw_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-woman, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.nakedw_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-woman, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.nakedw_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-woman, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.nakedm_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-man, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.nakedm_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-man, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.nakedm_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-man, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.nakedm_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['dressed person']" \
                    MACE.multi_concept="[ [ [naked-man, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
8 experiments generated
['mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.nakedw_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.nakedw_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.nakedw_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.nakedw_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.nakedm_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.nakedm_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.nakedm_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.nakedm_sd1.4.bf16_r0']



50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mmouse_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cartoon']" \
                    MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.mmouse_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cartoon']" \
                    MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mmouse_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cartoon']" \
                    MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.mmouse_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cartoon']" \
                    MACE.multi_concept="[ [ [mickey-mouse, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.r2d2_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['robot']" \
                    MACE.multi_concept="[ [ [r2d2-robot, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.r2d2_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['robot']" \
                    MACE.multi_concept="[ [ [r2d2-robot, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.r2d2_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['robot']" \
                    MACE.multi_concept="[ [ [r2d2-robot, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.r2d2_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['robot']" \
                    MACE.multi_concept="[ [ [r2d2-robot, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.gcat_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cat']" \
                    MACE.multi_concept="[ [ [grumpy-cat, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.gcat_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cat']" \
                    MACE.multi_concept="[ [ [grumpy-cat, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.gcat_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cat']" \
                    MACE.multi_concept="[ [ [grumpy-cat, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.gcat_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['cat']" \
                    MACE.multi_concept="[ [ [grumpy-cat, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.macbook_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['laptop']" \
                    MACE.multi_concept="[ [ [macbook, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.macbook_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['laptop']" \
                    MACE.multi_concept="[ [ [macbook, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.macbook_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['laptop']" \
                    MACE.multi_concept="[ [ [macbook, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
50
 python mace/training.py mace/configs/custom/erase_custom_1.yaml \
                    exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.macbook_sd1.4.bf16_r0" \
                    MACE.base_output_dir="data_root2/logs/mace" \
                    MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
                    MACE.use_gsam_mask='true' use_sam_hq='true' \
                    MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
                    MACE.rank=1 MACE.num_gen_images=8 \
                    MACE.prior_preservation_cache_path="" \
                    MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
                    MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
                    MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
                    MACE.mapping_concept="['laptop']" \
                    MACE.multi_concept="[ [ [macbook, object] ] ]" \
                    MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
16 experiments generated
['mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mmouse_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.mmouse_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mmouse_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.mmouse_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.r2d2_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.r2d2_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.r2d2_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.r2d2_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.gcat_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.gcat_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.gcat_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.gcat_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.macbook_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.macbook_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.macbook_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.macbook_sd1.4.bf16_r0']


#  python mace/training.py mace/configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.4CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.4CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.4CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.4CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.4CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.4CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.4CELEB00_sd1.4.bf16_r0']


# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.8CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.8CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.8CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.8CELEB00_sd1.4.bf16_r0']

#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.4CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.4CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 3 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.4CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.4CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.4CELEB00_sd1.4.bf16_r0']


# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.8CELEB00_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.mapping_concept="['a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person']" \
#                     MACE.multi_concept="[[['margot-robbie', 'object'], ['david-beckham', 'object'], ['barack-obama', 'object'], ['rihanna', 'object'], ['emma-stone', 'object'], ['elon-musk', 'object'], ['morgan-freeman', 'object'], ['oprah-winfrey', 'object']]]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 3 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.8CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.8CELEB00_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.8CELEB00_sd1.4.bf16_r0']


# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2.5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.rihanna_sd1.4.bf16_r0']

# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 1 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+5.as0_U.naked_sd1.4.bf16_r0']
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.pollock_sd1.4.bf16_r0']

# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.rihanna_sd1.4.bf16_r0']



# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.pollock_sd1.4.bf16_r0']


# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=5e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=2e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb2e+4_U.naked_sd1.4.bf16_r0']
# # 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 8 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.naked_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.naked_sd1.4.bf16_r0']

# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0']


# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.picasso_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.pollock_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.pollock_sd1.4.bf16_r0']

# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0']
# # 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0']

# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 50
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+0_U.rihanna_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2_U.rihanna_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4_U.rihanna_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6_U.rihanna_sd1.4.bf16_r0']
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+2 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=1e+6 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+0.as0_U.rihanna_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+2.as0_U.rihanna_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+4.as0_U.rihanna_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb1e+6.as0_U.rihanna_sd1.4.bf16_r0']
# is it because of the domain cache is suck
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb1.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=1.0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb4.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb32.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=32 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=100 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg8e+2.tr1e0.fr1e0.lamb1.00.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+2.tr1e0.fr1e0.lamb4.00.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+2.tr1e0.fr1e0.lamb32.00.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0']
# # 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg2e+2.tr1e0.fr1e0.lamb1.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=2e+2 \
#                     MACE.lamb=1.0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg2e+2.tr1e0.fr1e0.lamb4.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=2e+2 \
#                     MACE.lamb=4 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg2e+2.tr1e0.fr1e0.lamb32.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=2e+2 \
#                     MACE.lamb=32 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 0
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg2e+2.tr1e0.fr1e0.lamb100.00.as0_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=0 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=2e+2 \
#                     MACE.lamb=100 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg2e+2.tr1e0.fr1e0.lamb1.00.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg2e+2.tr1e0.fr1e0.lamb4.00.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg2e+2.tr1e0.fr1e0.lamb32.00.as0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg2e+2.tr1e0.fr1e0.lamb1000.00.as0_U.mrobbie_sd1.4.bf16_r0']
# amb0.50.attns0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg3e+2.tr1e0.fr1e0.lamb0.50.attns0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg4e+2.tr1e0.fr1e0.lamb0.50.attns0_U.mrobbie_sd1.4.bf16_r0', 'mace.psg6e+2.tr1e0.fr1e0.lamb0.50.attns0_U.mrobbie_sd1.4.bf16_r0']
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+4 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+4 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+4 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+2 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+4 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+2.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+4.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+6.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+8.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0']



#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+6 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+6.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+7.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+8.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+9.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e-1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+0.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+1.tr1e0.fr1e0.lamb0.50_U.rihanna_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.obama_sd1.4.bf16_r0', 'mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.mrobbie_sd1.4.bf16_r0', 'mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.beckham_sd1.4.bf16_r0', 'mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.rihanna_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_artist.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.picasso_sd1.4.bf16_r0', 'mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.vgogh_sd1.4.bf16_r0', 'mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.cmonet_sd1.4.bf16_r0', 'mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.pollock_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=0e+0 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="data_root/cache/mace/cache_coco.pt" \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path="data_root/cache/mace/custom/cache_dressed_person.pt"  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.lamb=0.0 MACE.fuse_preserve_scale=1e-4 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.psg0e+0.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0', 'mace.psg8e+1.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0', 'mace.psg8e+3.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0', 'mace.psg8e+5.coco.tr1e-4.fr1e-4.lamb0.00_U.naked_sd1.4.bf16_r0']

#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0.0 \
#                     MACE.lamb=0.5 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.tr1e0.fr1e0.lamb1.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0.0 \
#                     MACE.lamb=1.0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.tr1e0.fr1e0.lamb2.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0.0 \
#                     MACE.lamb=2.0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=""  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=0.0 \
#                     MACE.lamb=4.0 MACE.fuse_preserve_scale=1e0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.tr1e0.fr1e0.lamb1.00_U.obama_sd1.4.bf16_r0', 'mace.tr1e0.fr1e0.lamb2.00_U.obama_sd1.4.bf16_r0', 'mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0's]
                    
# 4 experiments generated
# ['mace.tr1e0.fr1e0.lamb0.50_U.obama_sd1.4.bf16_r0', 'mace.tr1e0.fr1e0.lamb1.00_U.obama_sd1.4.bf16_r0', 'mace.tr1e0.fr1e0.lamb2.00_U.obama_sd1.4.bf16_r0', 'mace.tr1e0.fr1e0.lamb4.00_U.obama_sd1.4.bf16_r0']
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="lamb0.5_generic8e+0_mace_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 





#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="lamb0.5_generic8e+1_mace_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 




#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="lamb0.5_generic8e+3_mace_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 





#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="lamb0.5_generic8e-1_mace_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="3p_mace.ps8e-1_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="testswapcoco_mace.ps8e-1_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e-1_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+1_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path="" \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e0 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.ps8e-1_U.obama_sd1.4.bf16_r0', 'mace.ps8e+1_U.obama_sd1.4.bf16_r0', 'mace.ps8e+3_U.obama_sd1.4.bf16_r0', 'mace.ps8e+5_U.obama_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+8.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+8.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+8.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+8 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.ps8e+8.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+8.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+8.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+8.coco1e-4_U.rihanna_sd1.4.bf16_r0']

#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.picasso_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#                     u
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.vgogh_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.cmonet_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='false' use_sam_hq='false' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['artist']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_artist.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.ps8e+3.coco1e-4_U.picasso_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.picasso_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.picasso_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.picasso_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.vgogh_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.vgogh_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.vgogh_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.vgogh_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.cmonet_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.cmonet_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.cmonet_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.cmonet_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.pollock_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.pollock_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.pollock_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.pollock_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask='true' use_sam_hq='true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.ps8e+3.coco1e-4_U.naked_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.naked_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.naked_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.naked_sd1.4.bf16_r0']


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask 'true' use_sam_hq 'true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask 'true' use_sam_hq 'true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask 'true' use_sam_hq 'true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.naked_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.use_gsam_mask 'true' use_sam_hq 'true' \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['dressed person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_dressed_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.ps8e+3.coco1e-4_U.naked_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.naked_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.naked_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.naked_sd1.4.bf16_r0']
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [naked-person, object] ] ]" \
#                         MACE.use_gsam_mask=true MACE.use_sam_hq=true \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
# 0 experiments generated
# []


#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [picasso, style] ] ]" \
#                         MACE.use_gsam_mask false use_sam_hq false \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [van-gogh, style] ] ]" \
#                         MACE.use_gsam_mask false use_sam_hq false \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [claude-monet, style] ] ]" \
#                         MACE.use_gsam_mask false use_sam_hq false \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_custom_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [jackson-pollock, style] ] ]" \
#                         MACE.use_gsam_mask false use_sam_hq false \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
# 0 experiments generated
# []


#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_custom_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 12 experiments generated
# ['mace.ps8e+3.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0']


# export CUDA_HOME=/usr/local/cuda-12.1/
# export CUDA_HOME=/usr/local/cuda-12/
#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 



#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+7 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root2/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+9 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.ps8e+3.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+7.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+9.coco1e-4_U.rihanna_sd1.4.bf16_r0']

#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
#  python data_preparation.py configs/custom/erase_celeb_1.yaml \
#                         exp_name="" \
#                         MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                         MACE.num_gen_images=8 MACE.seed=2024 \
#                         MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                         MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 



#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e-1.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+1.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e-1.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+1.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.mrobbie_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [margot-robbie, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e-1.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+1.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.beckham_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [david-beckham, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e-1.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+1.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [rihanna, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 16 experiments generated
# ['mace.ps8e-1.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+1.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e-1.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+1.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.mrobbie_sd1.4.bf16_r0', 'mace.ps8e-1.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+1.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.beckham_sd1.4.bf16_r0', 'mace.ps8e-1.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+1.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.rihanna_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.rihanna_sd1.4.bf16_r0']




#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e-1.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e-1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+1.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+1 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+3.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+3 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
#  python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps8e+5.coco1e-4_U.obama_sd1.4.bf16_r0" \
#                     MACE.base_output_dir="data_root/logs/mace" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 MACE.num_gen_images=8 \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.mapping_concept="['a person']" \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt  \
#                     MACE.train_preserve_scale=1e-4 MACE.preserve_weight=8e+5 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 
                    
# 4 experiments generated
# ['mace.ps8e-1.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+1.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+3.coco1e-4_U.obama_sd1.4.bf16_r0', 'mace.ps8e+5.coco1e-4_U.obama_sd1.4.bf16_r0']


#        python training.py configs/custom/erase_celeb_1.yaml \
#                     exp_name="mace.ps0coco_U.obama_sd1.4.bf16_r0" \
#                     MACE.pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
#                     MACE.learning_rate=1e-4 MACE.max_train_steps=50 MACE.seed=2024 \
#                     MACE.rank=1 \
#                     MACE.num_gen_images=8 \
#                     MACE.domain_preservation_cache_path=data_root/cache/mace/custom/cache_person.pt MACE.mapping_concept="['a person']" \
#                     MACE.prior_preservation_cache_path=data_root/cache/mace/cache_coco.pt \
#                     MACE.train_preserve_scale=0 MACE.preserve_weight=0 \
#                     MACE.multi_concept="[ [ [barack-obama, object] ] ]" \
#                     MACE.input_data_dir="data_root/generated/mace/sd1.4/r0" 

# CUDA_VISIBLE_DEVICES=0 python inference.py \
#           --num_images 3 \
#           --prompt 'a photo of Barack Obama' \
#           --model_path data_root/logs/mace.ps0coco_U.obama_sd1.4.bf16_r0/LoRA_fusion_model \
#           --save_path data_root/generated/study/test_mace_ps0coco0_from_mace




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