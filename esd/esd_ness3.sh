
export device="cuda:3"

python esd_sd_surgery.py --erase_concept 'mackerel tabby cat' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 3000 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --batch_size 4 --seed 0
python esd_sd_surgery.py --erase_concept 'mackerel tabby cat' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 3000 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --batch_size 4 --seed 0



# python esd_sd_surgery.py --erase_concept 'a painting in the style of Van Gogh' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 3000 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --preservation_weight 0.80 --preservation_train_set '00' --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --base_concept 'general'  --erase_from 'general'  --special_log_step '0,100,200,300,400'  --batch_size 4 --seed 0
# ['esd-x.nG1.00.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0', 'esd-x.nG3.00.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0']
# Total experiments: 2

# python esd_sd_surgery.py --erase_concept 'Dolnald Trump' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 1500 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 0.00 --unlearn_proj_prob 0.50  --erase_from 'uncond'  --special_log_step '0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475'  --batch_size 4 --seed 0
# python esd_sd_surgery.py --erase_concept 'Dolnald Trump' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 1500 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 1.00 --unlearn_proj_prob 0.50  --erase_from 'uncond'  --special_log_step '0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475'  --batch_size 4 --seed 0
# python esd_sd_surgery.py --erase_concept 'Dolnald Trump' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 1500 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 2.00 --unlearn_proj_prob 0.50  --erase_from 'uncond'  --special_log_step '0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475'  --batch_size 4 --seed 0
# python esd_sd_surgery.py --erase_concept 'Dolnald Trump' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 1500 --log_step 250 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --unlearn_proj_prob 0.50  --erase_from 'uncond'  --special_log_step '0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475'  --batch_size 4 --seed 0
# ['esd-x.fU_U.dtrump_sd1.4.bf16.bs4_r0', 'esd-x.nG1.00.fU_U.dtrump_sd1.4.bf16.bs4_r0', 'esd-x.nG2.00.fU_U.dtrump_sd1.4.bf16.bs4_r0', 'esd-x.nG3.00.fU_U.dtrump_sd1.4.bf16.bs4_r0']
# Total experiments: 4


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x'  --lr 5e-5 --save_path '../data_root/logs/esd/study/' --max_training_step 1500 --log_step 20 --device $device  --train_precision 'bf16'  --negative_guidance 3.00 --preservation_weight 0.20 --preservation_train_set '00' --preservation_weight_option 'convex'  --unlearn_proj_prob 0.50  --extra_forward_prob 0.50 --forward_general  --forward_preserve  --extra_forward_negative_guidance 0.00  --erase_from 'uncond'  --special_log_step '0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480'  --batch_size 4 --seed 0 --use_indiv_extra_forward --test_tag 'indiv_forward'
