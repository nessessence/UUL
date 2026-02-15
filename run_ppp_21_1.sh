

export device="cuda:1"

python ppp/reo_ppp.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.20 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
  --train_data_dir 'data_root/generated/stereo/a photo of Margot Robbie/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
  --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 1 --ang_push_pair_option 'allpair' --use_erase_ti_weight 

# python ppp/train_ppp_erase.py --erase_concept 'Barack Obama' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'Barack Obama' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'Barack Obama' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'Barack Obama' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0


# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Van Gogh' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Van Gogh' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Van Gogh' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Van Gogh' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0

# python ppp/train_ppp_erase.py --erase_concept 'naked person' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0



# python ppp/train_ppp_erasing.py --erase_concept 'naked person' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/test/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.80 --ang_incl_margin ex0.80 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0
