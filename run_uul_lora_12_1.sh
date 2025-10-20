export CUDA_VISIBLE_DEVICES=1
export pc_id="12_1"



            accelerate launch train_dreambooth_lora.py \
                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-ganesha-from-ganesha-esdx.safetensors" \
                --load_lora_weight_path="" \
                --instance_data_dir="data_root/data/real_data/dummy" \
                --gen_image_path="data_root/generated/model/esd-x.ganesha_sd1.4" \
                --output_dir="data_root/logs/gen" \
                --validation_prompt="a photo of festival dance;a photo of candle flame;a photo of pilgrimage;a photo of ritual fire;a photo of data center" --instance_prompt="a photo of festival dance;a photo of candle flame;a photo of pilgrimage;a photo of ritual fire;a photo of data center" \
                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                --run_note 'gen img' --wait_weight \
                --num_validation_images 50 \
                --cfg_scale 7.50 --gen_batch 10


#            accelerate launch train_dreambooth_lora.py \
#                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                --load_unet_weight_path="" \
#                --load_lora_weight_path="" \
#                --instance_data_dir="data_root/data/real_data/dummy" \
#                --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
#                --output_dir="data_root/logs/gen" \
#                --validation_prompt="a photo of idol;a photo of stupa;a photo of spiritual symbol;a photo of puja ritual;a photo of Kartikeya" --instance_prompt="a photo of idol;a photo of stupa;a photo of spiritual symbol;a photo of puja ritual;a photo of Kartikeya" \
#                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                --run_note 'gen img' --wait_weight \
#                --num_validation_images 50 \
#                --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1


# echo 'count:0 - sd1.4 0 /'

#        accelerate launch train_dreambooth_lora.py \
#            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#            --load_unet_weight_path="" \
#            --load_lora_weight_path="" \
#            --instance_data_dir="data_root/data/real_data/dummy" \
#            --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
#            --output_dir="data_root/logs/gen" \
#            --validation_prompt="a photo of Megan Fox;a photo of Anne Hathaway;a photo of Scarlett Johansson;a photo of Blake Lively;a photo of Natalie Portman;a photo of Amber Heard;a photo of Cameron Diaz;a photo of Emily Blunt;a photo of Angelina Jolie;a photo of Keira Knightley" --instance_prompt="a photo of Megan Fox;a photo of Anne Hathaway;a photo of Scarlett Johansson;a photo of Blake Lively;a photo of Natalie Portman;a photo of Amber Heard;a photo of Cameron Diaz;a photo of Emily Blunt;a photo of Angelina Jolie;a photo of Keira Knightley" \
#            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#            --run_note 'gen img' --wait_weight \
#            --num_validation_images 50 \
#            --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1


# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.pad_thai_sd1.4']
# echo 'count:0 - thai_sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-pad_thai-from-pad_thai-esdx.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.pad_thai_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of fluorescent lights;a photo of sandals;a photo of lotus pond;a photo of tea whisk;a photo of sled;a photo of japanese shrine;a photo of iron skillet;a photo of busy street stall;a photo of planet rings;a photo of fried tofu cubes" --instance_prompt="a photo of fluorescent lights;a photo of sandals;a photo of lotus pond;a photo of tea whisk;a photo of sled;a photo of japanese shrine;a photo of iron skillet;a photo of busy street stall;a photo of planet rings;a photo of fried tofu cubes" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --cfg_scale 7.50 --gen_batch 10
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

# echo 'count:0 - sd1.4 0 /'

#            accelerate launch train_dreambooth_lora.py \
#                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                --load_unet_weight_path="" \
#                --load_lora_weight_path="" \
#                --instance_data_dir="data_root/data/real_data/dummy" \
#                --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
#                --output_dir="data_root/logs/gen" \
#                --validation_prompt="a photo of studio lights;a photo of black and white cartoon;a photo of Road Runner;a photo of Energizer Bunny;a photo of cartoon parade;a photo of storybook illustration;a photo of sunset beach;a photo of sandbox;a photo of poisoned apple" --instance_prompt="a photo of studio lights;a photo of black and white cartoon;a photo of Road Runner;a photo of Energizer Bunny;a photo of cartoon parade;a photo of storybook illustration;a photo of sunset beach;a photo of sandbox;a photo of poisoned apple" \
#                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                --run_note 'gen img' --wait_weight \
#                --num_validation_images 50 \
#                --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 1


# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.rihanna_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG1.00.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG1.00.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a blonde-haired woman with wavy hair;a photo of a woman with blue eyes and fair skin;a photo of a glamorous actress on the red carpet;a photo of a woman in pink outfit with stylish makeup;a photo of a close-up portrait of smiling woman with red lipstick;a photo of a fashionable woman in a designer evening gown;a photo of a young woman with sharp jawline and defined cheekbones;a photo of a woman with shoulder-length hair styled in soft curls;a photo of a actress posing for a magazine photoshoot;a photo of a confident woman in a tailored suit with elegant posture" --instance_prompt="a photo of a blonde-haired woman with wavy hair;a photo of a woman with blue eyes and fair skin;a photo of a glamorous actress on the red carpet;a photo of a woman in pink outfit with stylish makeup;a photo of a close-up portrait of smiling woman with red lipstick;a photo of a fashionable woman in a designer evening gown;a photo of a young woman with sharp jawline and defined cheekbones;a photo of a woman with shoulder-length hair styled in soft curls;a photo of a actress posing for a magazine photoshoot;a photo of a confident woman in a tailored suit with elegant posture" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a blonde-haired woman with wavy hair;a photo of a woman with blue eyes and fair skin;a photo of a glamorous actress on the red carpet;a photo of a woman in pink outfit with stylish makeup;a photo of a close-up portrait of smiling woman with red lipstick;a photo of a fashionable woman in a designer evening gown;a photo of a young woman with sharp jawline and defined cheekbones;a photo of a woman with shoulder-length hair styled in soft curls;a photo of a actress posing for a magazine photoshoot;a photo of a confident woman in a tailored suit with elegant posture" --instance_prompt="a photo of a blonde-haired woman with wavy hair;a photo of a woman with blue eyes and fair skin;a photo of a glamorous actress on the red carpet;a photo of a woman in pink outfit with stylish makeup;a photo of a close-up portrait of smiling woman with red lipstick;a photo of a fashionable woman in a designer evening gown;a photo of a young woman with sharp jawline and defined cheekbones;a photo of a woman with shoulder-length hair styled in soft curls;a photo of a actress posing for a magazine photoshoot;a photo of a confident woman in a tailored suit with elegant posture" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a blonde-haired woman with wavy hair;a photo of a woman with blue eyes and fair skin;a photo of a glamorous actress on the red carpet;a photo of a woman in pink outfit with stylish makeup;a photo of a close-up portrait of smiling woman with red lipstick;a photo of a fashionable woman in a designer evening gown;a photo of a young woman with sharp jawline and defined cheekbones;a photo of a woman with shoulder-length hair styled in soft curls;a photo of a actress posing for a magazine photoshoot;a photo of a confident woman in a tailored suit with elegant posture" --instance_prompt="a photo of a blonde-haired woman with wavy hair;a photo of a woman with blue eyes and fair skin;a photo of a glamorous actress on the red carpet;a photo of a woman in pink outfit with stylish makeup;a photo of a close-up portrait of smiling woman with red lipstick;a photo of a fashionable woman in a designer evening gown;a photo of a young woman with sharp jawline and defined cheekbones;a photo of a woman with shoulder-length hair styled in soft curls;a photo of a actress posing for a magazine photoshoot;a photo of a confident woman in a tailored suit with elegant posture" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 3


# ['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.00.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.rihanna_sd1.4']
# echo 'count:0 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG1.00.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG1.00.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a tall muscular man;a photo of a man with long blonde hair;a photo of a man with short blonde hair and a beard;a photo of a man with blue eyes;a photo of a man in a superhero costume;a photo of a man wearing medieval armor;a photo of a man holding a hammer;a photo of a australian actor on the red carpet;a photo of a man with a chiseled jawline;a photo of a man in a tailored suit at a movie premiere" --instance_prompt="a photo of a tall muscular man;a photo of a man with long blonde hair;a photo of a man with short blonde hair and a beard;a photo of a man with blue eyes;a photo of a man in a superhero costume;a photo of a man wearing medieval armor;a photo of a man holding a hammer;a photo of a australian actor on the red carpet;a photo of a man with a chiseled jawline;a photo of a man in a tailored suit at a movie premiere" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:1 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a tall muscular man;a photo of a man with long blonde hair;a photo of a man with short blonde hair and a beard;a photo of a man with blue eyes;a photo of a man in a superhero costume;a photo of a man wearing medieval armor;a photo of a man holding a hammer;a photo of a australian actor on the red carpet;a photo of a man with a chiseled jawline;a photo of a man in a tailored suit at a movie premiere" --instance_prompt="a photo of a tall muscular man;a photo of a man with long blonde hair;a photo of a man with short blonde hair and a beard;a photo of a man with blue eyes;a photo of a man in a superhero costume;a photo of a man wearing medieval armor;a photo of a man holding a hammer;a photo of a australian actor on the red carpet;a photo of a man with a chiseled jawline;a photo of a man in a tailored suit at a movie premiere" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:2 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.rihanna_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a tall muscular man;a photo of a man with long blonde hair;a photo of a man with short blonde hair and a beard;a photo of a man with blue eyes;a photo of a man in a superhero costume;a photo of a man wearing medieval armor;a photo of a man holding a hammer;a photo of a australian actor on the red carpet;a photo of a man with a chiseled jawline;a photo of a man in a tailored suit at a movie premiere" --instance_prompt="a photo of a tall muscular man;a photo of a man with long blonde hair;a photo of a man with short blonde hair and a beard;a photo of a man with blue eyes;a photo of a man in a superhero costume;a photo of a man wearing medieval armor;a photo of a man holding a hammer;a photo of a australian actor on the red carpet;a photo of a man with a chiseled jawline;a photo of a man in a tailored suit at a movie premiere" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 3

"""


echo 'count:0 - sd1.4 0 /'

       accelerate launch train_dreambooth_lora.py \
           --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
           --load_unet_weight_path="" \
           --load_lora_weight_path="" \
           --instance_data_dir="data_root/data/real_data/dummy" \
           --gen_image_path="data_root/generated/model/original_pretrained_sd1.4" \
           --output_dir="data_root/logs/gen" \
           --validation_prompt="a photo of a caribbean woman;a photo of a pop singer performing on stage;a photo of a fashion model in haute couture dress;a photo of a young woman with almond-shaped eyes and high cheekbones;a photo of a curly black hair with volume;a photo of a confident woman with bold red lipstick;a photo of a singer holding a microphone under spotlight;a photo of a celebrity in glamorous evening gown;a photo of a woman with stylish short pixie haircut;a photo of a confident woman with expressive stage presence" --instance_prompt="a photo of a caribbean woman;a photo of a pop singer performing on stage;a photo of a fashion model in haute couture dress;a photo of a young woman with almond-shaped eyes and high cheekbones;a photo of a curly black hair with volume;a photo of a confident woman with bold red lipstick;a photo of a singer holding a microphone under spotlight;a photo of a celebrity in glamorous evening gown;a photo of a woman with stylish short pixie haircut;a photo of a confident woman with expressive stage presence" \
           --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
           --run_note 'gen img' --wait_weight \
           --num_validation_images 50 \
           --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
           --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 1



['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.cat_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.dog_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.horse_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.cow_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.car_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.bus_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.motorcycle_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.bicycle_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.mountain_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.river_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.forest_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.desert_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-cat-from-cat-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.cat_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-dog-from-dog-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.dog_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-horse-from-horse-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.horse_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-cow-from-cow-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.cow_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-car-from-car-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.car_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bus-from-bus-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.bus_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-motorcycle-from-motorcycle-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.motorcycle_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bicycle-from-bicycle-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.bicycle_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-mountain-from-mountain-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.mountain_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-river-from-river-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.river_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-forest-from-forest-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.forest_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-desert-from-desert-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.desert_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12


['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.obama_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.edsheeran_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.mrobbie_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.chemsworth_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.cevans_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.aadam_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.ahathaway_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.mcarey_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.octavia_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.morganf_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-car-from-car-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black woman" --instance_prompt="a photo of a black woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12


echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-car-from-car-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.car_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-motorcycle-from-motorcycle-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.motorcycle_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bicycle-from-bicycle-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.bicycle_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" --instance_prompt="a photo of Barrack Obama;a photo of Rihanna;a photo of Ed Sheeran;a photo of Margot Robbie;a photo of Chris Hemsworth;a photo of Chris Evans;a photo of Amy Adams;a photo of Anne Hathaway;a photo of Mariah Carey;a photo of Octavia Spencer;a photo of Morgan Freeman;a photo of Drake" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 6



echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-car-from-car-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.car_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-motorcycle-from-motorcycle-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.motorcycle_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-bicycle-from-bicycle-esdx_nG1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.00.bicycle_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 6



        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-forest-from-forest-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.forest_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle;a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cat;a photo of a dog;a photo of a horse;a photo of a mountain;a photo of a car;a photo of a motorcycle;a photo of a bicycle;a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'



['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.obama_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.edsheeran_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.mrobbie_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.chemsworth_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.cevans_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.aadam_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.ahathaway_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.mcarey_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.octavia_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.morganf_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.obama_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.edsheeran_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.mrobbie_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.chemsworth_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" --instance_prompt="a photo of a black person;a photo of a white person;a photo of a asian person;a photo of a black man;a photo of a white man;a photo of a asian man;a photo of a asian woman;a photo of a white man;a photo of a white woman" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10






['rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.obama_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.rihanna_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.edsheeran_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.mrobbie_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.chemsworth_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.cevans_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.aadam_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.ahathaway_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.mcarey_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.octavia_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.morganf_sd1.4', 'rlct4.reV.dummy.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-car-from-car-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" --instance_prompt="a photo of a cow;a photo of a bus;a photo of a mountain;a photo of a river;a photo of a forest;a photo of a desert" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12





# echo 'count:7 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.ahathaway_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a monkey" --instance_prompt="a photo of a monkey" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:8 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.mcarey_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a monkey" --instance_prompt="a photo of a monkey" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:9 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.octavia_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a monkey" --instance_prompt="a photo of a monkey" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:10 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.morganf_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a monkey" --instance_prompt="a photo of a monkey" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:11 - sd1.4 0 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG3.00_PS1.00_BGeneral.safetensors" \
#             --load_lora_weight_path="" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.BGeneral.drake_sd1.4" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of a monkey" --instance_prompt="a photo of a monkey" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --cfg_scale 7.50 --gen_batch 10
# Total scripts generated: 12
['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of a cat;a photo of a car;a photo of a mountain" --instance_prompt="a photo of a cat;a photo of a car;a photo of a mountain" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12




['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG1.50.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG1.50.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG1.50.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
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
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG3.00_PS1.00.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.PS1.00.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12




['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.nG3.00.BGeneral.drake_sd1.4']
echo 'count:0 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.obama_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.rihanna_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.edsheeran_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.mrobbie_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.chemsworth_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.cevans_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.aadam_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.ahathaway_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.mcarey_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.octavia_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.morganf_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_nG3.00_BGeneral.safetensors" \
            --load_lora_weight_path="" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="data_root/generated/model/esd-x.nG3.00.BGeneral.drake_sd1.4" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --donot_reinit_validation_generator \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 12



echo 'count:0 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/asante/aligned/asante-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/reese/aligned/reese-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/nivola/aligned/nivola-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/earle/aligned/earle-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/leowoodal/aligned/leowoodal-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/starkey/aligned/starkey-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/apierre/aligned/apierre-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/skyhblack/aligned/skyhblack-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/sophiewilde/aligned/sophiewilde-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/edebiri/aligned/edebiri-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/mmadison/aligned/mmadison-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --instance_data_dir="data_root/data/real_data/nicoparker/aligned/nicoparker-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4" \
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
['rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4', 'rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4', 'rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4', 'rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4', 'rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4', 'rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4', 'rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4', 'rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4', 'rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4', 'rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4', 'rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4', 'rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4']
echo 'count:0 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:13 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:14 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:15 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:16 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:17 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:18 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:19 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:20 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:21 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mrobbie_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:22 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:23 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:24 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:25 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:26 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:27 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:28 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:30 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:31 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:32 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:33 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:34 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:44 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:45 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:46 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:47 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:48 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:49 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:50 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:51 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:52 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:53 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:54 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.cevans_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:55 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:56 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:57 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:58 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:59 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:60 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:61 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:62 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:63 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:64 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:65 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:66 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:67 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:68 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:69 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:70 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:71 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:72 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:73 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:74 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:75 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:76 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:77 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:78 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:79 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:80 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:81 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:82 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:83 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:84 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:85 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:86 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:87 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:88 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:89 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:90 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:91 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:92 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:93 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:94 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:95 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:96 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:97 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:98 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:99 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:100 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:101 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:102 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:103 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.mcarey_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.obama_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:121 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:122 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:123 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:124 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:125 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:126 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:127 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:128 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:129 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:130 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:131 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_nG3.00.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.nG3.00.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 132
#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-100" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-100" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:123 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4 200 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-200" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-200" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:124 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4 300 /'

#         accelerate launch train_dreambooth_lora.py \
#             --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#             --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
#             --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-300" \
#             --instance_data_dir="data_root/data/real_data/dummy" \
#             --gen_image_path="auto" \
#             --output_dir="data_root/logs/gen" \
#             --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
#             --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#             --run_note 'gen img' --wait_weight \
#             --num_validation_images 50 \
#             --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-300" \
#             --placeholder_token="v1" --initializer_token='person' \
#             --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
#             --cfg_scale 7.50 --gen_batch 10
# echo 'count:125 - rlct4.reV.nicoparkerA5V0.l

echo 'count:103 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.morganf_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.mrobbie_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:121 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:122 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:123 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:124 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:125 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:126 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:127 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:128 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:129 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:130 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:131 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_dTavg_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4_esd-x.BGeneral.dTavg.s500.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 132


echo 'count: 0'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/obama/aligned/obama-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul obamaA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 1'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/rihanna/aligned/rihanna-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul rihannaA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 2'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/edsheeran/aligned/edsheeran-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul edsheeranA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 3'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mrobbie/aligned/mrobbie-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul mrobbieA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 4'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/chemsworth/aligned/chemsworth-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul chemsworthA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 5'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/cevans/aligned/cevans-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul cevansA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 6'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/aadam/aligned/aadam-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul aadamA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 7'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/ahathaway/aligned/ahathaway-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul ahathawayA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 8'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/mcarey/aligned/mcarey-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul mcareyA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 9'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/octavia/aligned/octavia-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul octaviaA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 10'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/morganf/aligned/morganf-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul morganfA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
echo 'count: 11'

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --instance_data_dir="data_root/data/real_data/drake/aligned/drake-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --train_batch_size=1 --gradient_accumulation_steps=4 \
            --lora_rank 4 --target_lora_modules to_k to_v --target_lora_layers cross \
            --max_train_steps=1000  --validation_steps=50  --checkpointing_steps=50  --lr_scheduler "linear"  --seed 2 \
            --run_note 'uul drakeA5V0 lNone ti' \
            --cfg_scale 6.0 \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --with_prior_preservation --prior_loss_weight=1.0 --num_class_images 200 \
            --class_prompt="a photo of a person" --class_data_dir="data_root/generated/model/original_pretrained_sd1.4/a photo of a person_neg/7.50" \
            --learning_rate_lora 1e-4 --learning_rate_ti 5e-4 \
            --train_text_encoder --learning_rate_lora_text_encoder 1e-5 \
            --placeholder_token="v1" --initializer_token='person'
['rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4', 'rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4', 'rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4', 'rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4', 'rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4', 'rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4', 'rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4', 'rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4', 'rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4', 'rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4', 'rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4', 'rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4']
echo 'count:0 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.obamaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.obama_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:13 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:14 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:15 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:16 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:17 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:18 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:19 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:20 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:21 - rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.rihannaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:22 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:23 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:24 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:25 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:26 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:27 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:28 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:30 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:31 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:32 - rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edsheeranA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:33 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:34 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mrobbieA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mrobbie_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:44 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:45 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:46 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:47 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:48 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:49 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:50 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:51 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:52 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:53 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:54 - rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Hemsworth-from-Chris_Hemsworth-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.chemsworthA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.chemsworth_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:55 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:56 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:57 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:58 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:59 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:60 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:61 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:62 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:63 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:64 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:65 - rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.cevansA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.cevans_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:66 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:67 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:68 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:69 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:70 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:71 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:72 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:73 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:74 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:75 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:76 - rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Amy_Adams-from-Amy_Adams-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.aadamA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.aadam_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:77 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:78 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:79 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:80 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:81 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:82 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:83 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:84 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:85 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:86 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:87 - rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.ahathawayA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:88 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:89 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:90 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:91 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:92 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:93 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:94 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:95 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:96 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:97 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:98 - rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mcareyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.mcarey_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:99 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:100 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:101 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:102 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:103 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Octavia_Spencer-from-Octavia_Spencer-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.octaviaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.octavia_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Morgan_Freeman-from-Morgan_Freeman-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.morganfA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.morganf_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:121 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:122 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:123 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:124 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:125 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:126 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:127 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:128 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:129 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:130 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:131 - rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Drake-from-Drake-esdx_BGeneral_step500.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.drakeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r2_esd-x.BGeneral.s500.drake_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 132

echo 'count:0 '

            accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/asante/aligned/asante-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/reese/aligned/reese-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/nivola/aligned/nivola-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/earle/aligned/earle-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/leowoodal/aligned/leowoodal-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/starkey/aligned/starkey-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/apierre/aligned/apierre-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/skyhblack/aligned/skyhblack-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/sophiewilde/aligned/sophiewilde-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/edebiri/aligned/edebiri-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/mmadison/aligned/mmadison-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4" \
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
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --instance_data_dir="data_root/data/real_data/nicoparker/aligned/nicoparker-5-v0" \
            --output_dir="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4" \
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
['rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4', 'rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4', 'rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4', 'rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4', 'rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4', 'rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4', 'rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4', 'rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4', 'rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4', 'rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4', 'rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4', 'rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4']
echo 'count:0 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:1 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:2 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:3 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:4 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:5 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:6 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:7 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:8 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:9 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:10 - rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.asanteA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:11 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:12 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:13 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:14 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:15 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:16 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:17 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:18 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:19 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:20 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:21 - rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Margot_Robbie-from-Margot_Robbie-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.reeseA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mrobbie_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:22 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:23 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:24 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:25 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:26 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:27 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:28 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:29 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:30 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:31 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:32 - rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nivolaA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:33 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:34 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:35 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:36 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:37 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:38 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:39 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:40 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:41 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:42 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:43 - rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.earleA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:44 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:45 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:46 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:47 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:48 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:49 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:50 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:51 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:52 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:53 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:54 - rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Chris_Evans-from-Chris_Evans-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.leowoodalA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.cevans_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:55 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:56 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:57 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:58 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:59 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:60 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:61 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:62 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:63 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:64 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:65 - rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.starkeyA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:66 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:67 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:68 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:69 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:70 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:71 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:72 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:73 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:74 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:75 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:76 - rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Rihanna-from-Rihanna-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.apierreA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.rihanna_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:77 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:78 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:79 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:80 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:81 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:82 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:83 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:84 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:85 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:86 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:87 - rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Anne_Hathaway-from-Anne_Hathaway-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.skyhblackA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.ahathaway_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:88 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:89 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:90 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:91 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:92 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:93 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:94 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:95 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:96 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:97 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:98 - rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.sophiewildeA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:99 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:100 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:101 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:102 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:103 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:104 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:105 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:106 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:107 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:108 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:109 - rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Mariah_Carey-from-Mariah_Carey-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.edebiriA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.mcarey_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:110 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:111 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:112 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:113 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:114 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:115 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:116 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:117 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:118 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:119 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:120 - rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Barrack_Obama-from-Barrack_Obama-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.mmadisonA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.obama_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:121 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 0 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-0" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-0" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:122 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 100 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-100" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-100" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:123 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 200 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-200" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-200" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:124 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 300 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-300" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-300" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:125 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 400 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-400" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-400" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:126 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 500 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-500" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-500" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:127 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 600 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-600" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-600" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:128 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 700 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-700" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-700" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:129 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 800 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-800" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-800" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:130 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 900 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-900" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-900" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
echo 'count:131 - rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4 1000 /'

        accelerate launch train_dreambooth_lora.py \
            --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
            --load_unet_weight_path="data_root/logs/esd/sd1.4/esd-Ed_Sheeran-from-Ed_Sheeran-esdx_dTavg.safetensors" \
            --load_lora_weight_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-1000" \
            --instance_data_dir="data_root/data/real_data/dummy" \
            --gen_image_path="auto" \
            --output_dir="data_root/logs/gen" \
            --validation_prompt="a photo of v1" --instance_prompt="a photo of v1" \
            --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
            --run_note 'gen img' --wait_weight \
            --num_validation_images 50 \
            --load_token_embedding_path="data_root/logs/rlct4.reV.nicoparkerA5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4.r1_esd-x.dTavg.edsheeran_sd1.4/checkpoint-1000" \
            --placeholder_token="v1" --initializer_token='person' \
            --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality." \
            --cfg_scale 7.50 --gen_batch 10
Total scripts generated: 132
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

