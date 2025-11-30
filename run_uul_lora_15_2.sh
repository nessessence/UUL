export CUDA_VISIBLE_DEVICES=2
export pc_id="15_2"

                        accelerate launch train_dreambooth_lora.py \
                            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                            --load_unet_weight_path="data_root/logs/esd/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/step400.safetensors" \
                            --load_lora_weight_path="" \
                            --instance_data_dir="data_root/data/real_data/dummy" \
                            --gen_image_path="data_root/generated/study/esd-x.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0_testfasttrack/step400" \
                            --output_dir="data_root/logs/gen" \
                            --validation_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" --instance_prompt="a photo of Margot Robbie;a photo of Jennifer Lawrence;a photo of Reese Witherspoon;a photo of Jessica Chastain;a photo of Gal Gadot;a photo of Brad Pitt;a photo of Kristen Stewart;a photo of Anne Hathaway;a photo of Leonardo DiCaprio;a photo of Meryl Streep;a photo of Nicole Kidman" \
                            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                            --run_note 'gen img' --wait_weight \
                            --num_validation_images 50 \
                            --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 



                        # accelerate launch train_dreambooth_lora.py \
                        #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                        #     --load_unet_weight_path="data_root/logs/esd/study/esd-x.fU_U.dtrump_sd1.4.bf16.bs4_r0/step300.safetensors" \
                        #     --load_lora_weight_path="" \
                        #     --instance_data_dir="data_root/data/real_data/dummy" \
                        #     --gen_image_path="data_root/generated/study/esd-x.fU_U.dtrump_sd1.4.bf16.bs4_r0_testfasttrack/step300" \
                        #     --output_dir="data_root/logs/gen" \
                        #     --validation_prompt="a photo of Donald Trump;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of he 45th president of the united states;a photo of the 45th president of the united states in disneyland;a photo of Donald Trump in disneyland;a photo of Donald Trump in a style of cartoon" --instance_prompt="a photo of Donald Trump;a photo of Joe Biden;a photo of candidate;a photo of Bernie Sanders;a photo of White House;a photo of Republican;a photo of Melania Trump;a photo of Barack Obama;a photo of politician;a photo of Mitch McConnell;a photo of Xi Jinping;a photo of he 45th president of the united states;a photo of the 45th president of the united states in disneyland;a photo of Donald Trump in disneyland;a photo of Donald Trump in a style of cartoon" \
                        #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                        #     --run_note 'gen img' --wait_weight \
                        #     --num_validation_images 50 \
                        #     --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 