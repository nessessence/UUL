
export device="cuda:3"


python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 1500 --log_step 20 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --preservation_weight 0.20 --preservation_train_set '00' --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --extra_forward_prob 0.50 --forward_general  --forward_preserve  --extra_forward_negative_guidance 0.00  --erase_from 'uncond'  --special_log_step '0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480'  --batch_size 4 --seed 0 --use_indiv_extra_forward --test_tag 'indiv_forward'
