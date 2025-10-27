
export device="cuda:2"

# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 0 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --seed 0 --max_training_step 500



# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 1 --max_training_step 500   
# python esd_sd_surgery.py --erases_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 2 --max_training_step 500




# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00   --seed 0 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00   --seed 1 --max_training_step 500
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00   --seed 2 --max_training_step 500


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 2 --max_training_step 500


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 0 --max_training_step 500 --preservation_train_set '00'  
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 1 --max_training_step 500 --preservation_train_set '00'  
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --seed 2 --max_training_step 500 --preservation_train_set '00'  


# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --seed 0 --max_training_step 500 --preservation_train_set '00'  
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --seed 1 --max_training_step 500 --preservation_train_set '00'  
# python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --seed 2 --max_training_step 500 --preservation_train_set '00' 




python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500 --preservation_train_set '00'
python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 2.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'



python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 0 --max_training_step 500 --preservation_train_set '00'
python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'attn_head' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'



python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 1 --max_training_step 500 --preservation_train_set '00'
python esd_sd_surgery.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/pg/' --device $device  --negative_guidance 3.00 --preservation_weight 1.00   --apply_gradient_projection --gradient_projection_mode 'hard' --gradient_projection_param_group 'global' --gradient_projection_preserve_scale 1.00  --seed 2 --max_training_step 500 --preservation_train_set '00'

# python esd_sd_ness.py --erase_concept 'nordic house' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'chimpanzee' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'pad thai' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# # python esd_sd_ness.py --erase_concept 'persian cat' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# # python esd_sd_ness.py --erase_concept 'claude monet' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# ['esd-nordic_house-from-nordic_house-esdx', 'esd-chimpanzee-from-chimpanzee-esdx', 'esd-pad_thai-from-pad_thai-esdx', 'esd-persian_cat-from-persian_cat-esdx', 'esd-claude_monet-from-claude_monet-esdx']

# python esd_sd_ness.py --erase_concept 'mickey mouse' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device

# ['esd-mickey_mouse-from-mickey_mouse-esdx', 'esd-mario-from-mario-esdx', 'esd-ganesha-from-ganesha-esdx']



# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --base_concept general --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG1.50_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_nG1.50_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG1.50_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG1.50_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG1.50_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG1.50_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG1.50_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG1.50_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG1.50_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG1.50_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG1.50_BGeneral', 'esd-Drake-from-Drake-esdx_nG1.50_BGeneral']



# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_PS1.00', 'esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_PS1.00', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_PS1.00', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_PS1.00', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_PS1.00', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_PS1.00', 'esd-Drake-from-Drake-esdx_nG3.00_PS1.00']


# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --preservation_weight 1.00 --base_concept general --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG2.50_PS1.00_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_nG2.50_PS1.00_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG2.50_PS1.00_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG2.50_PS1.00_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG2.50_PS1.00_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG2.50_PS1.00_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG2.50_PS1.00_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG2.50_PS1.00_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG2.50_PS1.00_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG2.50_PS1.00_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG2.50_PS1.00_BGeneral', 'esd-Drake-from-Drake-esdx_nG2.50_PS1.00_BGeneral']


# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG1.00', 'esd-Rihanna-from-Rihanna-esdx_nG1.00', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG1.00', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG1.00', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG1.00', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG1.00', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG1.00', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG1.00', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG1.00', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG1.00', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG1.00', 'esd-Drake-from-Drake-esdx_nG1.00']


# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --base_concept general --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_PS1.00_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_PS1.00_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_PS1.00_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_PS1.00_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_PS1.00_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_PS1.00_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_PS1.00_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_PS1.00_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_PS1.00_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_PS1.00_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_PS1.00_BGeneral', 'esd-Drake-from-Drake-esdx_PS1.00_BGeneral']


# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_PS1.00_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_PS1.00_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_PS1.00_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_PS1.00_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_PS1.00_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_PS1.00_BGeneral', 'esd-Drake-from-Drake-esdx_nG3.00_PS1.00_BGeneral']


# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device


# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral', 'esd-Drake-from-Drake-esdx_BGeneral']
# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# python esd_sd.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 0-500 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_T0-500', 'esd-Rihanna-from-Rihanna-esdx_T0-500', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_T0-500', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_T0-500', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_T0-500', 'esd-Chris_Evans-from-Chris_Evans-esdx_T0-500', 'esd-Amy_Adams-from-Amy_Adams-esdx_T0-500', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_T0-500', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_T0-500', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_T0-500', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_T0-500', 'esd-Drake-from-Drake-esdx_T0-500']