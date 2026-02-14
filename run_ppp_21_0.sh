

export device="cuda:0"
export CUDA_VISIBLE_DEVICES=0


python ppp/train_ppp_erase.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
python ppp/train_ppp_erase.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
python ppp/train_ppp_erase.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
python ppp/train_ppp_erase.py --erase_concept 'Margot Robbie' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0



python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Picasso' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Picasso' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Picasso' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0
python ppp/train_ppp_erase.py --erase_concept 'a painting in the style of Picasso' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0


python ppp/train_ppp_erase.py --erase_concept 'naked person' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/study/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.50 --unlearn_proj_prob 0.50  --batch_size 4 --seed 0



# python ppp/train_ppp_erasing.py --erase_concept 'naked person' --train_method 'esd-x-kv'  --lr 5e-5 --save_path 'data_root/logs/esd/test/' --max_training_step 1000 --log_step 1000 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --preservation_weight 1.00 --preservation_train_set 'UG' --preservation_weight_option 'additive'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --timestep_constraint '999-1000'  --aei_loss_weight 1.00 --ang_excl_margin 0.80 --ang_incl_margin ex0.80 --sim_param_group 'attn_head' --ang_norm_loss_weight 0.00 --generic_loss_weight 0.00 --ang_incl_loss_weight 1.00 --ang_excl_loss_weight 1.00  --ang_preserve_loss_weight 32.00  --batch_size 4 --seed 0
