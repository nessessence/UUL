

export device="cuda:3"

python ppp/reo_ppp.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.20 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
  --train_data_dir 'data_root/generated/stereo/a photo of Margot Robbie/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
    --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 3 --ang_push_pair_option 'allpair' --ang_all_pair_aggr_concept_option 'avg' --use_erase_ti_weight  --final_ang_preserve_loss_weight 500.00 --skip_stage1

['esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.20I0.60P32.00-N0.00G0.00_U.mrobbie_sd1.4.bf16.bs4_r0']
Total experiments: 1

# python ppp/reo_ppp.py --erase_concept 'David Beckham' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.20 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
#   --train_data_dir 'data_root/generated/stereo/a photo of David Beckham/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
#     --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 3 --ang_push_pair_option 'allpair' --use_erase_ti_weight   --ang_all_pair_aggr_concept_option 'max'




# sleep 5m
# python ppp/reo_ppp.py --erase_concept 'David Beckham' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.20 --ang_incl_margin 0.8 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
#   --train_data_dir 'data_root/generated/stereo/a photo of David Beckham/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
#     --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 1  --use_erase_ti_weight --skip_stage1


# python ppp/reo_ppp.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.20 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
#   --train_data_dir 'data_root/generated/stereo/a photo of Margot Robbie/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
#     --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 1  --use_erase_ti_weight --skip_stage1


# python ppp/reo_ppp.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.20 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
#   --train_data_dir 'data_root/generated/stereo/a photo of Rihanna/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
#     --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 3 

# python ppp/reo_ppp.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.40 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
#   --train_data_dir 'data_root/generated/stereo/a photo of Rihanna/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
#     --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 3 

# python ppp/reo_ppp.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.60 --ang_incl_margin 0.6 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0 \
#   --train_data_dir 'data_root/generated/stereo/a photo of Rihanna/' --learnable_property 'object' --initializer_token 'person' --load_erased_weight_if_exist \
#     --ti_used_train_steps 500 --ti_max_train_steps 500 --total_ti_attack_iterations 3 
