# IMG_DIR1=image_save_folder
# IMG_DIR2=image_save_folder
# MODE=evaluation_mode ## kid / fid / KID / FID

# CUDA_VISIBLE_DEVICES=0 python ./metrics/evaluate_fid_score.py \
#     --dir1 "data_root/generated/Moodeng_0-99" \
#     --dir2 "data_root/generated/Moodeng_100-199" \
#     --mode kid 





CUDA_VISIBLE_DEVICES=1 python ./metrics/evaluate_fid_score.py \
    --dir1 "/home/nessessence/uul/data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-1000/gen_images/A photo of a v1" \
    --dir2 "data_root/generated/Moodeng" \
    --mode kid 
0.015357629959959977



CUDA_VISIBLE_DEVICES=1 python ./metrics/evaluate_fid_score.py \
    --dir1 "/home/nessessence/uul/data_root/logs/c.l4.kv_moodeng3-V_f0.5_lr.ti1e-2.l5e-5_b1g4/checkpoint-500/gen_images/A photo of a v1" \
    --dir2 "data_root/generated/Moodeng" \
    --mode kid 


    # CUDA_VISIBLE_DEVICES=0 python metrics/evaluate_fid.py --dir1 'path/to/generated/image/folder' --dir2 'path/to/coco/GT/folder'