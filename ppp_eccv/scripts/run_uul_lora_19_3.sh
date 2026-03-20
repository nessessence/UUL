export CUDA_VISIBLE_DEVICES=3
export pc_id="19_3"



                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/stereo/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie" --instance_prompt="a photo of Margot Robbie" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root2/logs/stereo/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 25 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


# echo 'count:0 - stereo-u.G2.00_U.obama_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root2/logs/stereo/stereo-u.G2.00_U.obama_sd1.4.bf16_r0/final_reo_unet.pt" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/stereo-u.G2.00_U.obama_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - stereo-u.G2.00_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root2/logs/stereo/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 25 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - stereo-u.G2.00_U.rihanna_sd1.4.bf16 0
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root2/logs/stereo/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/stereo-u.G2.00_U.rihanna_sd1.4.bf16_r0/step0" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
$$$$
: << 'COMMENT'
echo 'count:0 - stereo-u_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/stereo/stereo-u_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo-u_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - stereo-u_U.rihanna_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/stereo/stereo-u_U.rihanna_sd1.4.bf16_r0/final_reo_unet.pt" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/stereo-u_U.rihanna_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Rihanna;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.rihanna_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
