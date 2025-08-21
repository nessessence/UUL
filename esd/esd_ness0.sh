
export device="cuda:0"
export CUDA_VISIBLE_DEVICES=0

python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' 
python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' 
python esd_sd.py --erase_concept 'Amy Adams' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' 


# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Barrack Obama' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Rihanna' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Ed Sheeran' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Margot Robbie' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Chris Hemsworth' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Chris Evans' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Adam Driver' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Adam Driver' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Adam Driver' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Andrew Garfield' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Andrew Garfield' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Andrew Garfield' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Anne Adam' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Anne Adam' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Anne Adam' --train_method 'esd-all' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-x' --save_path '../data_root/logs/esd/sd1.4/' --device $device
# python esd_sd.py --erase_concept 'Anne Hathaway' --train_method 'esd-u' --save_path '../data_root/logs/esd/sd1.4/' --device $device
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