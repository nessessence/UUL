# IMG_DIR1=image_save_folder
# IMG_DIR2=image_save_folder
# MODE=evaluation_mode ## kid / fid / KID / FID

# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "data_root/generated/Moodeng_0-99" \
#     --dir2 "data_root/generated/Moodeng_100-199" \
#     --mode kid 


# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "data_root/generated/hippo" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# # 0.05064776155905919

# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "/home/nessessence/uul/data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images/A photo of a v1" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# # 0.015357779992492515


# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "/home/nessessence/uul/data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images/A photo of a v1" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# # 0.037657895477978015



# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "/home/nessessence/uul/data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500/gen_images/A photo of a v1" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# # KID score: 0.0236007925350349

# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "/home/nessessence/uul/data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1500/gen_images/A photo of a v1" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# KID score: 0.05292858266016014


# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "/home/nessessence/uul/data_root/logs/uul_moodeng.object_c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images/A photo of a v1" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# # 0.037325145265265106


# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "/home/nessessence/uul/data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images/A photo of a v1" \
#     --dir2 "data_root/generated/Moodeng" \
#     --mode kid 
# # 0.04366873583583568
# ####




CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
    --dir1 "/home/nessessence/uul/data_root/logs/uul_moodeng.object_c.l16.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images/A photo of a v1" \
    --dir2 "data_root/generated/Moodeng" \
    --mode kid 
# KID score: 0.12385081245245244 **

    # CUDA_VISIBLE_DEVICES=0 python metrics/evaluate_fid.py --dir1 'path/to/generated/image/folder' --dir2 'path/to/coco/GT/folder'