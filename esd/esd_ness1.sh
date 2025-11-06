
export device="cuda:1"



python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/stats/' --max_training_step 1000  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00 --preservation_train_set '00' --preservation_weight 0.50 --preservation_weight_option 'convex'  --unlearn_proj_prob 1.00  --seed 0 --collect_gradient_statistics_option 'static'





# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.00 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.10 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.20 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.30 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.40 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.50 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.60 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.70 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.80 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.90 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 1
# ['esd-x.nG3.00_GP.gG.pH-u0.50.pe00-PS0.00_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.10_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.20_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.30_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.40_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.50_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.60_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.70_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.80_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS0.90_U.mrobbie_sd1.4_r1', 'esd-x.nG3.00_GP.gG.pH-u0.50.pe00-cPS1.00_U.mrobbie_sd1.4_r1']


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 300  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 0.80 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 300  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 0.90 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 300  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --seed 0

# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/test/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.90 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.10 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.20 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.30 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.40 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.50 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.60 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.70 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.80 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.90 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 1.00 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2



# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.10 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.20 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.30 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.40 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.50 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.60 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.70 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.80 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.90 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 1.00 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 2
# ['esd-x.nG3.00_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.10_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.20_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.30_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.40_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.50_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.60_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.70_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.80_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.90_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS1.00_U.mrobbie_sd1.4_r2', 'esd-x.nG3.00_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.10_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.20_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.30_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.40_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.50_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.60_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.70_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.80_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS0.90_U.ganesha_sd1.4_r2', 'esd-x.nG3.00.pe00-cPS1.00_U.ganesha_sd1.4_r2']


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.10 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.20 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.30 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.40 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.50 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.60 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.70 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.80 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 0.90 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --preservation_weight 1.00 --preservation_train_set '00' --preservation_weight_option 'convex'  --seed 0
# ['esd-x.nG3.00.pe00-cPC0.10_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.20_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.30_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.40_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.50_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.60_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.70_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.80_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC0.90_U.mrobbie_sd1.4_r0', 'esd-x.nG3.00.pe00-cPC1.00_U.mrobbie_sd1.4_r0']

# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.10 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.20 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.30 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.40 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.50 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.60 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.70 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.80 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 0.90 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 3.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'base' --gradient_projection_preserve_scale 1.00 --preservation_train_set '00' --preservation_weight 1.00 --preservation_weight_option 'convex'  --seed 0
# ['esd-x.nG3.00_GP.gB.pH.pe00-cPS0.10_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.20_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.30_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.40_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.60_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.70_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.80_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS0.90_U.ganesha_sd1.4_r0', 'esd-x.nG3.00_GP.gB.pH.pe00-cPS1.00_U.ganesha_sd1.4_r0']

# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.10 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.20 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.30 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.40 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.50 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.60 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.70 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.80 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.90 --preservation_train_set '00'  --seed 0
# ['esd-x.nG2.00.pe00-PS0.10_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.20_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.30_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.40_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.50_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.60_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.70_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.80_U.mrobbie_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.90_U.mrobbie_sd1.4_r0']


# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.10 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.20 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.30 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.40 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.50 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.60 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.70 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.80 --preservation_train_set '00'  --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --preservation_weight 0.90 --preservation_train_set '00'  --seed 0
# ['esd-x.nG2.00.pe00-PS0.10_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.20_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.30_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.40_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.50_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.60_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.70_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.80_U.ganesha_sd1.4_r0', 'esd-x.nG2.00.pe00-PS0.90_U.ganesha_sd1.4_r0']


# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.10 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.20 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.30 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.40 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.50 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.60 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.70 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.80 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# python esd_sd_surgery.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --max_training_step 500  --device $device  --negative_guidance 2.00 --apply_gradient_projection --gradient_projection_mode 'hard'  --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 0.90 --preservation_train_set '00' --preservation_weight 1.00 --seed 0
# ['esd-x.nG2.00_GP.gG.pH.pe00-PS0.10_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.20_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.30_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.40_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.50_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.60_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.70_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.80_U.ganesha_sd1.4_r0', 'esd-x.nG2.00_GP.gG.pH.pe00-PS0.90_U.ganesha_sd1.4_r0']
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'



# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'




# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'



# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500



# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00   --seed 0 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00   --seed 1 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00   --seed 2 --max_training_step 500


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500



# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# export CUDA_VISIBLE_DEVICES=1

# python _esd_sd_surgery_attn.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4_/'  --preservation_weight 1.00 --base_concept general  --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00 --device $device

# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4_/'  --preservation_weight 1.00 --base_concept general  --device $device
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4_/'  --preservation_weight 1.00 --base_concept general  --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00 --device $device
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4_/'  --preservation_weight 1.00 --base_concept general  --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'neuron' --gradient_projection_preserve_scale 1.00 --device $device

# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4_/'  --preservation_weight 1.00 --base_concept general  --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00 --device $device




# python esd_sd_ness.py --erase_concept 'cat' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'dog' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'horse' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'car' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'motorcycle' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'bicycle' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# ['esd-cat-from-cat-esdx', 'esd-dog-from-dog-esdx', 'esd-horse-from-horse-esdx', 'esd-car-from-car-esdx', 'esd-motorcycle-from-motorcycle-esdx', 'esd-bicycle-from-bicycle-esdx']



# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device

# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --max_training_step 500  --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_dTavg_step500', 'esd-Rihanna-from-Rihanna-esdx_BGeneral_dTavg_step500', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_dTavg_step500', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_dTavg_step500', 'esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_dTavg_step500', 'esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_dTavg_step500', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_dTavg_step500', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_dTavg_step500', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500', 'esd-Drake-from-Drake-esdx_BGeneral_dTavg_step500']


# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000  --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_T750-1000', 'esd-Rihanna-from-Rihanna-esdx_T750-1000', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_T750-1000', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_T750-1000', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_T750-1000', 'esd-Chris_Evans-from-Chris_Evans-esdx_T750-1000', 'esd-Amy_Adams-from-Amy_Adams-esdx_T750-1000', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_T750-1000', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_T750-1000', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_T750-1000', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_T750-1000', 'esd-Drake-from-Drake-esdx_T750-1000']



# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_T0-750', 'esd-Rihanna-from-Rihanna-esdx_T0-750', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_T0-750', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_T0-750', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_T0-750', 'esd-Chris_Evans-from-Chris_Evans-esdx_T0-750', 'esd-Amy_Adams-from-Amy_Adams-esdx_T0-750', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_T0-750', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_T0-750', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_T0-750', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_T0-750', 'esd-Drake-from-Drake-esdx_T0-750']




# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 500-1000  --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_T500-1000', 'esd-Rihanna-from-Rihanna-esdx_T500-1000', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_T500-1000', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_T500-1000', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_T500-1000', 'esd-Chris_Evans-from-Chris_Evans-esdx_T500-1000', 'esd-Amy_Adams-from-Amy_Adams-esdx_T500-1000', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_T500-1000', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_T500-1000', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_T500-1000', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_T500-1000', 'esd-Drake-from-Drake-esdx_T500-1000']


# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 250-1000 
# ['esd-Barrack_Obama-from-Barrack_Obama-esdu_T250-1000', 'esd-Rihanna-from-Rihanna-esdu_T250-1000', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T250-1000', 'esd-Margot_Robbie-from-Margot_Robbie-esdu_T250-1000', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T250-1000', 'esd-Chris_Evans-from-Chris_Evans-esdu_T250-1000', 'esd-Amy_Adams-from-Amy_Adams-esdu_T250-1000', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T250-1000', 'esd-Mariah_Carey-from-Mariah_Carey-esdu_T250-1000', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T250-1000', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdu_T250-1000', 'esd-Drake-from-Drake-esdu_T250-1000']


# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 
# ['esd-Barrack_Obama-from-Barrack_Obama-esdu_T750-1000', 'esd-Rihanna-from-Rihanna-esdu_T750-1000', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T750-1000', 'esd-Margot_Robbie-from-Margot_Robbie-esdu_T750-1000', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T750-1000', 'esd-Chris_Evans-from-Chris_Evans-esdu_T750-1000', 'esd-Amy_Adams-from-Amy_Adams-esdu_T750-1000', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T750-1000', 'esd-Mariah_Carey-from-Mariah_Carey-esdu_T750-1000', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T750-1000', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdu_T750-1000', 'esd-Drake-from-Drake-esdu_T750-1000']





# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750


# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 



# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-750  --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdu', 'esd-Rihanna-from-Rihanna-esdu', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdu', 'esd-Margot_Robbie-from-Margot_Robbie-esdu', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu', 'esd-Chris_Evans-from-Chris_Evans-esdu', 'esd-Amy_Adams-from-Amy_Adams-esdu', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdu', 'esd-Mariah_Carey-from-Mariah_Carey-esdu', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdu', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdu', 'esd-Drake-from-Drake-esdu']

# # python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# # python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# # python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-250 
# ['esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-250', 'esd-Rihanna-from-Rihanna-esdu_T0-250', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-250', 'esd-Margot_Robbie-from-Margot_Robbie-esdu_T0-250', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-250', 'esd-Chris_Evans-from-Chris_Evans-esdu_T0-250', 'esd-Amy_Adams-from-Amy_Adams-esdu_T0-250', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-250', 'esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-250', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-250', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdu_T0-250', 'esd-Drake-from-Drake-esdu_T0-250']


# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500  --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdu_T0-500', 'esd-Rihanna-from-Rihanna-esdu_T0-500', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T0-500', 'esd-Margot_Robbie-from-Margot_Robbie-esdu_T0-500', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T0-500', 'esd-Chris_Evans-from-Chris_Evans-esdu_T0-500', 'esd-Amy_Adams-from-Amy_Adams-esdu_T0-500', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T0-500', 'esd-Mariah_Carey-from-Mariah_Carey-esdu_T0-500', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T0-500', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdu_T0-500', 'esd-Drake-from-Drake-esdu_T0-500']

# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdu_T750-1000', 'esd-Rihanna-from-Rihanna-esdu_T750-1000', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdu_T750-1000', 'esd-Margot_Robbie-from-Margot_Robbie-esdu_T750-1000', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdu_T750-1000', 'esd-Chris_Evans-from-Chris_Evans-esdu_T750-1000', 'esd-Amy_Adams-from-Amy_Adams-esdu_T750-1000', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdu_T750-1000', 'esd-Mariah_Carey-from-Mariah_Carey-esdu_T750-1000', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdu_T750-1000', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdu_T750-1000', 'esd-Drake-from-Drake-esdu_T750-1000']



# # python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Adam Driver' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Adam Driver' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Adam Driver' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Andrew Garfield' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Andrew Garfield' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Andrew Garfield' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Anne Adam' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Anne Adam' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Anne Adam' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# # python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Angelina Jolie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Angelina Jolie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Angelina Jolie' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Amber Heard' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Amber Heard' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Amber Heard' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Oprah Winfrey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Oprah Winfrey' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Oprah Winfrey' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Idris Elba' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Idris Elba' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Idris Elba' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device