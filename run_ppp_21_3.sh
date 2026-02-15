

export device="cuda:3"



# python ppp/train_ppp_erase.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'Rihanna' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# ['esd-x-kv.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG0.50_U.obama_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.obama_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.obama_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.obama_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG0.50_U.beckham_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.beckham_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.beckham_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.beckham_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG0.50_U.rihanna_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.rihanna_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.rihanna_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.rihanna_sd1.4.bf16.bs4_r0']
# Total experiments: 16




# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Jackson Pollock' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Jackson Pollock' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Jackson Pollock' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Jackson Pollock' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# ['esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG0.50_U.vgogh_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.vgogh_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.vgogh_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.vgogh_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG0.50_U.cmonet_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.cmonet_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.cmonet_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.cmonet_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG0.50_U.pollock_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.pollock_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.pollock_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.pollock_sd1.4.bf16.bs4_r0']
# Total experiments: 16


# python ppp/train_ppp_erase.py --erase_concept 'naked person' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
# ['esd-x-kv.nG0.50_U.naked_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG1.00_U.naked_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG2.00_U.naked_sd1.4.bf16.bs4_r0', 'esd-x-kv.nG3.00_U.naked_sd1.4.bf16.bs4_r0']
# Total experiments: 4
# python ppp/train_ppp_erasing.py --erase_concept 'naked person' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/test/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.80 --ang_incl_margin ex0.80 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0
