export CUDA_VISIBLE_DEVICES=2

  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 512 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 




"""


# sleep 3h
            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG3.00_GP.gG.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0/step300.safetensors" \
            #     --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gG.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0.uS300/checkpoint-300" \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG3.00_GP.gG.pH.pe00-cPS0.50_U.ganesha_sd1.4_r0.uS300/checkpoint-300" \
            #     --placeholder_token="v1" --initializer_token='object' \
            #     --cfg_scale 7.50 --gen_batch 10



# export pc_id="21_2"
#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_U.ganesha_sd1.4_r1/step0.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG2.00_U.ganesha_sd1.4_r1.uS0/checkpoint-500" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG2.00_U.ganesha_sd1.4_r1.uS0/checkpoint-500" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10

            # accelerate launch train_dreambooth_lora.py \
            #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            #     --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_U.ganesha_sd1.4_r0/step0.safetensors" \
            #     --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_U.ganesha_sd1.4_r0.uS0/checkpoint-400" \
            #     --instance_data_dir="data_root/data/real_data/dummy" \
            #     --gen_image_path="auto" \
            #     --output_dir="data_root/logs/gen" \
            #     --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            #     --run_note 'gen img' --wait_weight \
            #     --num_validation_images 50 \
            #     --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r0_esd-x.nG2.00_U.ganesha_sd1.4_r0.uS0/checkpoint-400" \
            #     --placeholder_token="v1" --initializer_token='object' \
            #     --cfg_scale 7.50 --gen_batch 10
# echo 'count:19 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300 800 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300/checkpoint-800" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:20 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300 900 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300/checkpoint-900" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# echo 'count:21 - rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300 1000 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2/step300.safetensors" \
#                 --load_lora_weight_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="auto" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --load_token_embedding_path="data_root/logs/rlct4.reV.ganeshaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS300/checkpoint-1000" \
#                 --placeholder_token="v1" --initializer_token='object' \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 22



# echo 'count:0 - GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2.uS200 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/pg/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2/step200.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nG2.00_GP.gH.pH.pe00-PS0.00_U.ganesha_sd1.4_r2/step200" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of ganesha;a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti" --instance_prompt="a photo of ganesha;a photo of Lakshmi;a photo of mandir;a photo of Krishna;a photo of sanskrit script;a photo of Hanuman;a photo of meditation;a photo of Buddha statue;a photo of rat;a photo of elephant ride;a photo of murti" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10




# echo 'count:0 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of cosmetic palette;a photo of film set;a photo of fur coat;a photo of pyramid;a photo of apple fruit;a photo of golf club;a photo of teapot;a photo of beaded gown;a photo of movie premiere;a photo of oil painting" --instance_prompt="a photo of cosmetic palette;a photo of film set;a photo of fur coat;a photo of pyramid;a photo of apple fruit;a photo of golf club;a photo of teapot;a photo of beaded gown;a photo of movie premiere;a photo of oil painting" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1



# echo 'count:0 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of fresh coconut;a photo of tempura shrimp;a photo of egg scramble;a photo of thai bamboo steamer;a photo of mango sticky rice;a photo of Thai night market;a photo of coconut milk;a photo of soy sauce bottle;a photo of rice bowl;a photo of open wok flame" --instance_prompt="a photo of fresh coconut;a photo of tempura shrimp;a photo of egg scramble;a photo of thai bamboo steamer;a photo of mango sticky rice;a photo of Thai night market;a photo of coconut milk;a photo of soy sauce bottle;a photo of rice bowl;a photo of open wok flame" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1



# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.ganesha_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-ganesha-from-ganesha-esdx.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.ganesha_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of cave painting;a photo of sun chariot;a photo of desert sand;a photo of silk cloth;a photo of video game console;a photo of Roman aqueduct;a photo of incense smoke;a photo of airplane;a photo of hymn book;a photo of mosaic floor" --instance_prompt="a photo of cave painting;a photo of sun chariot;a photo of desert sand;a photo of silk cloth;a photo of video game console;a photo of Roman aqueduct;a photo of incense smoke;a photo of airplane;a photo of hymn book;a photo of mosaic floor" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1


# echo 'count:0 - house_sd1.4 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-nordic_house-from-nordic_house-esdx.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.nordic_house_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of Viking hall;a photo of lake cabin;a photo of ice hut;a photo of northern lights;a photo of fjordside cabin;a photo of Viking ship;a photo of Nordic kitchenware;a photo of bamboo house;a photo of fishing cabin;a photo of mountain chalet" --instance_prompt="a photo of Viking hall;a photo of lake cabin;a photo of ice hut;a photo of northern lights;a photo of fjordside cabin;a photo of Viking ship;a photo of Nordic kitchenware;a photo of bamboo house;a photo of fishing cabin;a photo of mountain chalet" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1





# ### from 15_1

# echo 'count:0 - mouse_sd1.4 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-mickey_mouse-from-mickey_mouse-esdx.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.mickey_mouse_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of studio lights;a photo of black and white cartoon;a photo of Road Runner;a photo of Energizer Bunny;a photo of cartoon parade;a photo of storybook illustration;a photo of sunset beach;a photo of sandbox;a photo of poisoned apple" --instance_prompt="a photo of studio lights;a photo of black and white cartoon;a photo of Road Runner;a photo of Energizer Bunny;a photo of cartoon parade;a photo of storybook illustration;a photo of sunset beach;a photo of sandbox;a photo of poisoned apple" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1

# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.ganesha_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-ganesha-from-ganesha-esdx.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.ganesha_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of festival dance;a photo of candle flame;a photo of pilgrimage;a photo of ritual fire;a photo of data center;a photo of cross;a photo of surfboard;a photo of tribal dance;a photo of garland;a photo of Burmese pagoda" --instance_prompt="a photo of festival dance;a photo of candle flame;a photo of pilgrimage;a photo of ritual fire;a photo of data center;a photo of cross;a photo of surfboard;a photo of tribal dance;a photo of garland;a photo of Burmese pagoda" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1
# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.ganesha_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-ganesha-from-ganesha-esdx.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.ganesha_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of idol;a photo of stupa;a photo of spiritual symbol;a photo of puja ritual;a photo of Kartikeya;a photo of swastika (Hindu);a photo of caravan camel;a photo of Himalayas;a photo of bull;a photo of lantern festival" --instance_prompt="a photo of idol;a photo of stupa;a photo of spiritual symbol;a photo of puja ritual;a photo of Kartikeya;a photo of swastika (Hindu);a photo of caravan camel;a photo of Himalayas;a photo of bull;a photo of lantern festival" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1
# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.ganesha_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#             accelerate launch train_dreambooth_lora.py \
#                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                 --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-ganesha-from-ganesha-esdx.safetensors" \
#                 --load_lora_weight_path="" \
#                 --instance_data_dir="data_root/data/real_data/dummy" \
#                 --gen_image_path="data_root/generated/model/esd-x.ganesha_sd1.4" \
#                 --output_dir="data_root/logs/gen" \
#                 --validation_prompt="a photo of elephant head;a photo of om symbol;a photo of temple elephant;a photo of murti;a photo of Lakshmi;a photo of Krishna;a photo of Durga;a photo of camel;a photo of Ramayana scene" --instance_prompt="a photo of elephant head;a photo of om symbol;a photo of temple elephant;a photo of murti;a photo of Lakshmi;a photo of Krishna;a photo of Durga;a photo of camel;a photo of Ramayana scene" \
#                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                 --run_note 'gen img' --wait_weight \
#                 --num_validation_images 50 \
#                 --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1


['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.cow_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.bus_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.mountain_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.river_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.forest_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.desert_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-cow-from-cow-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.cow_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle;a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle;a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bus-from-bus-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.bus_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle;a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle;a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'






['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.cow_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.bus_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.mountain_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.river_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.forest_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.desert_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-cow-from-cow-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.cow_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bus-from-bus-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.bus_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-mountain-from-mountain-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.mountain_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-river-from-river-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.river_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-forest-from-forest-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.forest_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-desert-from-desert-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.desert_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 6



['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.cow_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.bus_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.mountain_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.river_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.forest_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.desert_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-cow-from-cow-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.cow_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bus-from-bus-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.bus_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-mountain-from-mountain-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.mountain_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-river-from-river-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.river_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-forest-from-forest-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.forest_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-desert-from-desert-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.desert_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 6
['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.cow_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.bus_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.mountain_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.river_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.forest_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.desert_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-cow-from-cow-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.cow_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bus-from-bus-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.bus_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-mountain-from-mountain-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.mountain_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-river-from-river-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.river_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-forest-from-forest-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.forest_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-desert-from-desert-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.desert_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 6

echo 'count:0 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 1



['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12
['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.BGeneral.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12



echo 'count:0 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/asante/aligned/asante-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul asanteA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:1 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/reese/aligned/reese-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul reeseA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:2 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/nivola/aligned/nivola-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul nivolaA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:3 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/earle/aligned/earle-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul earleA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:4 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/leowoodal/aligned/leowoodal-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul leowoodalA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:5 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/starkey/aligned/starkey-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul starkeyA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:6 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/apierre/aligned/apierre-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul apierreA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:7 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/skyhblack/aligned/skyhblack-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul skyhblackA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:8 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/sophiewilde/aligned/sophiewilde-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul sophiewildeA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:9 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/edebiri/aligned/edebiri-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul edebiriA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:10 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/mmadison/aligned/mmadison-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul mmadisonA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count:11 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --instance_data_dir="data_root/data/real_data/nicoparker/aligned/nicoparker-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 1 \
            --run_note 'uul nicoparkerA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
12
['rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4', 'rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4', 'rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4', 'rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4', 'rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4', 'rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4', 'rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4', 'rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4', 'rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4', 'rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4', 'rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4', 'rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4']
echo 'count:0 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:13 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:14 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:15 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:16 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:17 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:18 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:19 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:20 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:21 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mrobbie_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:22 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:23 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:24 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:25 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:26 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:27 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:28 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:30 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:31 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:32 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:33 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:34 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:44 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:45 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:46 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:47 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:48 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:49 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:50 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:51 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:52 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:53 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:54 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.cevans_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:55 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:56 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:57 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:58 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:59 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:60 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:61 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:62 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:63 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:64 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:65 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:66 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:67 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:68 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:69 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:70 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:71 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:72 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:73 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:74 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:75 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:76 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:77 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:78 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:79 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:80 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:81 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:82 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:83 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:84 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:85 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:86 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:87 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:88 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:89 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:90 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:91 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:92 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:93 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:94 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:95 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:96 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:97 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:98 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:99 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:100 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:101 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:102 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:103 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.mcarey_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.obama_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:121 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:122 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:123 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:124 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:125 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:126 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:127 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:128 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:129 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:130 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:131 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 132

# echo 'count:0 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.obama_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.rihanna_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.edsheeran_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.mrobbie_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.chemsworth_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.cevans_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10


        # accelerate launch train_dreambooth_lora.py \
        #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        #     --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
        #     --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-1000" \
        #     --instance_data_dir="data_root/data/real_data/dummy" \
        #     --gen_image_path="auto" \
        #     --output_dir="data_root/logs/gen" \
        #     --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
        #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
        #     --run_note 'gen img' --wait_weight \
        #     --num_validation_images 50 \
        #     --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-1000" \
        #     --placeholder_token="v1" --initializer_token='person' \
        #     --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
        #     --cfg_scale 7.50 --gen_batch 10


        #     echo 'count:130 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4 900 /'

        # accelerate launch train_dreambooth_lora.py \
        #     --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
        #     --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
        #     --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-900" \
        #     --instance_data_dir="data_root/data/real_data/dummy" \
        #     --gen_image_path="auto" \
        #     --output_dir="data_root/logs/gen" \
        #     --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
        #     --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
        #     --run_note 'gen img' --wait_weight \
        #     --num_validation_images 50 \
        #     --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-900" \
        #     --placeholder_token="v1" --initializer_token='person' \
        #     --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
        #     --cfg_scale 7.50 --gen_batch 10




# echo 'count:110 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-0" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-0" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:111 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4 100 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:112 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10



['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10




            
# sleep 3h
# ['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.drake_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.obama_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.edsheeran_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.chemsworth_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.cevans_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.aadam_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:7 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.dTavg.s500.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12

# ['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.s500.drake_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.obama_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.edsheeran_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:3 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.mrobbie_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:4 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.chemsworth_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:5 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.cevans_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:6 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.aadam_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.BGeneral.s500.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12


['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.PS1.00.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
"""

  accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="data_root/data/real_data/moodeng/moodeng-unseen-3" \
  --output_dir="data_root/logs/noone" \
  --validation_prompt="A photo of a moodeng" \
  --instance_prompt="A photo of a moodeng" \
  --learning_rate=1e-4 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --lora_rank 512 \
  --test_run \
  --max_train_steps=10000000 --checkpointing_steps=100000000 --validation_steps=100000000 

