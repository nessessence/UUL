
export device="cuda:1"
# export CUDA_VISIBLE_DEVICES=1

python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000  --device $device
python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Mariah Carey' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Octavia Spencer' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Morgan Freeman' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
python esd_sd.py --erase_concept 'Drake' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --timestep_constraint 750-1000 --device $device
['esd-Barrack_Obama-from-Barrack_Obama-esdx_T750-1000', 'esd-Rihanna-from-Rihanna-esdx_T750-1000', 'esd-Ed_Sheeran-from-Ed_Sheeran-esdx_T750-1000', 'esd-Margot_Robbie-from-Margot_Robbie-esdx_T750-1000', 'esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_T750-1000', 'esd-Chris_Evans-from-Chris_Evans-esdx_T750-1000', 'esd-Amy_Adams-from-Amy_Adams-esdx_T750-1000', 'esd-Anne_Hathaway-from-Anne_Hathaway-esdx_T750-1000', 'esd-Mariah_Carey-from-Mariah_Carey-esdx_T750-1000', 'esd-Octavia_Spencer-from-Octavia_Spencer-esdx_T750-1000', 'esd-Morgan_Freeman-from-Morgan_Freeman-esdx_T750-1000', 'esd-Drake-from-Drake-esdx_T750-1000']



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