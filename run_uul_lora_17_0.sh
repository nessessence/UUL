export CUDA_VISIBLE_DEVICES=0
export pc_id="17_0"
echo 'count:0 - U.pddog_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" --instance_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.pddog_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" --instance_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.pddog_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" --instance_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.pddog_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" --instance_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.pddog_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" --instance_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.pddog_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.pddog_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" --instance_prompt="a photo of poodle dog;a photo of dog;a photo of beagle puppy;a photo of beagle dog running;a photo of golden retriever dog;a photo of persian cat;a photo of cat;a photo of dog;a photo of bichon frise dog;a photo of poodle dog;a photo of afghan hound dog;a photo of greyhound dog;a photo of dalmatian dog;a photo of mexican hairless dog" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
"""
echo 'count:0 - U.dtrump_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.dtrump_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.dtrump_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.dtrump_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.dtrump_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.dtrump_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG3.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG3.00.bG.fG_U.dtrump_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" --instance_prompt="a photo of Donald Trump;a photo of person;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of Kamala Harris;a photo of Margot Robbie;a photo of Morgan Freeman;a photo of Christ Hemsworth;a photo of Joe Biden" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.naked_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG1.00.bG.fG_U.naked_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" --instance_prompt="a photo of naked person;a photo of person;a photo of naked man;a photo of naked woman;a photo of dressed person;a photo of dressed man;a photo of dressed woman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

echo 'count:0 - U.ganesha_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.ganesha_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.ganesha_sd1.4.bf16.bs4_r0/step500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of ganesha;a photo of statue;a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti;a photo of statue of a god;a photo of elephant statue;a photo of hindu god statue" --instance_prompt="a photo of ganesha;a photo of statue;a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti;a photo of statue of a god;a photo of elephant statue;a photo of hindu god statue" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - U.cmonet_sd1.4.bf16.bs4_r0.uS1500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step1500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.cmonet_sd1.4.bf16.bs4_r0.uS2000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step2000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step2000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.cmonet_sd1.4.bf16.bs4_r0.uS2500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step2500.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step2500" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - U.cmonet_sd1.4.bf16.bs4_r0.uS3000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step3000.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.bG.fG_U.cmonet_sd1.4.bf16.bs4_r0/step3000" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" --instance_prompt="a painting in the style of Claude Monet;a painting in the style of artist;a painting in the style of Géricault;a painting in the style of Pollock;a painting in the style of Sisley;a painting in the style of Alma-Tadema;a painting in the style of Feininger;a painting in the style of Zorn;a painting in the style of Pissarro;a painting in the style of Ensor;a painting in the style of Bonnard;a painting in the style of Rivera;a painting in the style of Claude Monet;a painting in the style of Van Gogh;a photo of a water lilies painting;a photo of a haystacks painting" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step500.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS500/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
            --load_pretrained_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/vgogh/aligned/vgogh-5-v0" \
            --output_dir="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross --mixed_precision 'bf16' \
            --max_train_steps=1000  --validation_steps=1001  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 0 \
            --run_note 'uul dummy lNone ti' \
            --cfg_scale 6.0 \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='random'
echo 'count:0 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 0
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-0" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:1 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 100
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-100" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:2 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 200
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-200" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:3 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 300
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-300" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:4 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 400
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-400" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:5 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 500
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-500" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:6 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 600
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-600" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:7 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 700
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-700" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:8 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 800
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-800" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:9 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 900
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-900" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:10 - rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000 1000
'
                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                            --load_lora_weight_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="auto" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --load_token_embedding_path="data_root/logs/rlct4.reR.vgoghA5V0.ln.lr1e-4.ti5e-4.b1g4.bf16.r0_esd-x.bG.fG.pe00-cPS0.80_U.vgogh_sd1.4.bf16.bs4_r0.uS1000/checkpoint-1000" \
                            --placeholder_token="v1" --initializer_token='' \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
"""