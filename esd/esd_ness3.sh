
export device="cuda:3"
python esd_sd_ness.py --erase_concept 'persian cat' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
python esd_sd_ness.py --erase_concept 'claude monet' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
python esd_sd_ness.py --erase_concept 'mario' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
python esd_sd_ness.py --erase_concept 'ganesha' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --device $device
# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --preservation_weight 1.00 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_PS1.00', 'esd-Rihanna-from-Rihanna-esdx_PS1.00', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_PS1.00', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_PS1.00', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_PS1.00', 'esd-Chris_Evans-from-Chris_Evans-esdx_PS1.00', 'esd-Amy_Adams-from-Amy_Adams-esdx_PS1.00', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_PS1.00', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_PS1.00', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_PS1.00', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_PS1.00', 'esd-Drake-from-Drake-esdx_PS1.00']


# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg' --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --decompositional_timestep_sampler 'avg'
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_dTavg', 'esd-Rihanna-from-Rihanna-esdx_BGeneral_dTavg', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_dTavg', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_dTavg', 'esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_dTavg', 'esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_dTavg', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_dTavg', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_dTavg', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg', 'esd-Drake-from-Drake-esdx_BGeneral_dTavg']



# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --preservation_weight 1.00 --base_concept general --decompositional_timestep_sampler 'avg'
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_PS1.00_BGeneral_dTavg', 'esd-Drake-from-Drake-esdx_nG3.00_PS1.00_BGeneral_dTavg']








# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.50 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG1.50', 'esd-Rihanna-from-Rihanna-esdx_nG1.50', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG1.50', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG1.50', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG1.50', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG1.50', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG1.50', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG1.50', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG1.50', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG1.50', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG1.50', 'esd-Drake-from-Drake-esdx_nG1.50']

# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 2.50 --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG2.50', 'esd-Rihanna-from-Rihanna-esdx_nG2.50', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG2.50', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG2.50', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG2.50', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG2.50', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG2.50', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG2.50', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG2.50', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG2.50', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG2.50', 'esd-Drake-from-Drake-esdx_nG2.50']
# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 3.00 --base_concept general --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_nG3.00_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_BGeneral', 'esd-Drake-from-Drake-esdx_nG3.00_BGeneral']


# python esd_sd_ness.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --negative_guidance 1.00 --base_concept general --device $device
# ['esd-Barrack_Obama-from-Barrack_Obama-esdx_nG1.00_BGeneral', 'esd-Rihanna-from-Rihanna-esdx_nG1.00_BGeneral', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG1.00_BGeneral', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_nG1.00_BGeneral', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG1.00_BGeneral', 'esd-Chris_Evans-from-Chris_Evans-esdx_nG1.00_BGeneral', 'esd-Amy_Adams-from-Amy_Adams-esdx_nG1.00_BGeneral', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG1.00_BGeneral', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_nG1.00_BGeneral', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG1.00_BGeneral', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG1.00_BGeneral', 'esd-Drake-from-Drake-esdx_nG1.00_BGeneral']

# python esd_sd_ness.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device
# python esd_sd_ness.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/'  --base_concept general --device $device


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

