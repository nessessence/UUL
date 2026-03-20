export CUDA_VISIBLE_DEVICES=0
export pc_id="18_0"
echo 'count:0 - duo-s.b100_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b100_U.mmouse_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b100_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b250_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b250_U.mmouse_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b250_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.mmouse_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


echo 'count:0 - duo-s.b100_U.r2d2_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b100_U.r2d2_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b100_U.r2d2_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b250_U.r2d2_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b250_U.r2d2_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b250_U.r2d2_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - duo-s.b1000_U.r2d2_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="data_root/logs/duo/duo-s.b1000_U.r2d2_sd1.4.bf16_r0/checkpoint-1000" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/duo-s.b1000_U.r2d2_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
                                
$$$$
: << 'COMMENT'
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.60Ir0.60P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - uce_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/uce/uce_U.mmouse_sd1.4.bf16_r0/unet_weight.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/uce_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.mmouse_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.mmouse_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P32.00-N0.00G0.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.40P32.00-N0.00G0.00-mte0.rs_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - uce_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/uce/uce_U.mmouse_sd1.4.bf16_r0/unet_weight.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/uce_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.mmouse_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.mmouse_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.mmouse_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.mmouse_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.mmouse_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - uce_U.mrobbie_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/uce/uce_U.mrobbie_sd1.4.bf16_r0/unet_weight.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/uce_U.mrobbie_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.mrobbie_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.mrobbie_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.mrobbie_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG0.50_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG0.50_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG1.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG1.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG2.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG2.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG3.00_U.mrobbie_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00_U.mrobbie_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" --instance_prompt="a photo of Margot Robbie;a photo of Maya Angelou;a photo of Gillian Anderson;a photo of Jim Morrison;a photo of Jennifer Connelly;a photo of Benicio Del Toro;a photo of Avril Lavigne;a photo of Aaron Paul;a photo of Bill Murray;a photo of Kim Jong Un;a photo of Justin Bieber;a photo of David Bowie;a photo of Barry Manilow;a photo of Judy Garland;a photo of Betty White;a photo of Denise Richards;a photo of Gal Gadot;a photo of Pierce Brosnan;a photo of Julianne Moore;a photo of David Tennant;a photo of Jackie Chan;a photo of Natalie Portman;a photo of Rachel Dratch;a photo of Liv Tyler;a photo of Gordon Ramsey;a photo of Patrick Stewart;a photo of Doris Day;a photo of Matthew Mcconaughey;a photo of Amy Schumer;a photo of Hayley Atwell;a photo of Niall Horan;a photo of Neil Degrasse Tyson;a photo of Heath Ledger;a photo of Kristen Stewart;a photo of Amy Poehler;a photo of Kirsten Dunst;a photo of Matt Damon;a photo of Joan Rivers;a photo of Bill Nye;a photo of Britney Spears;a photo of Lizzy Caplan;a photo of Emma Roberts;a photo of Clint Eastwood;a photo of Rachel Mcadams;a photo of Harry Dean Stanton;a photo of Krysten Ritter;a photo of Aretha Franklin;a photo of Kate Upton;a photo of George Takei;a photo of Christina Hendricks;a photo of Andy Samberg" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.00-0.00P32.00-N0.00G0.00-mte0.rs_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.00-0.00P32.00-N0.00G0.00-mte0.rs_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
                                
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir-1.00P32.00-N0.00G0.00-mte0.rs_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir-1.00P32.00-N0.00G0.00-mte0.rs_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir-1.00P32.00-N0.00G0.00-mte0.rs_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 50 \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Ir0.60P32.00-N0.00G0.00-mte0.rs_U.aadam_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Ir0.60P32.00-N0.00G0.00-mte0.rs_U.aadam_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AhE0.40Ir0.60P32.00-N0.00G0.00-mte0.rs_U.aadam_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Amy Adams;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Amy Adams;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - uce_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/uce/uce_U.picasso_sd1.4.bf16_r0/unet_weight.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/uce_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.picasso_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.60Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P32.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P128.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.80Ir0.80P512.00-N0.00G0.00-mte0.rs_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - uce_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/uce/uce_U.picasso_sd1.4.bf16_r0/unet_weight.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/uce_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - suma-x.tie50_U.picasso_sd1.4.bf16 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root2/logs/suma/suma-x.tie50_U.picasso_sd1.4.bf16_r0/unet_weight_suma_final.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/suma-x.tie50_U.picasso_sd1.4.bf16_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of Abed Abdi;a painting in the style of Albert Marquet;a painting in the style of Alberto Vargas;a painting in the style of Adrian Tomine;a painting in the style of Albert Edelfelt;a painting in the style of A.J. Casson;a painting in the style of Abbott Fuller Graves;a painting in the style of Alan Kenny;a painting in the style of Alberto Sughi;a painting in the style of Aaron Siskind;a painting in the style of Adriaen van Outrecht;a painting in the style of Albert Goodwin;a painting in the style of Alberto Biasi;a painting in the style of Albert Bloch;a painting in the style of Adrian Donoghue;a painting in the style of Akos Major;a painting in the style of Alberto Burri;a painting in the style of Ai Weiwei;a painting in the style of Alastair Magnaldo;a painting in the style of Albert Watson;a painting in the style of Agnes Martin;a painting in the style of Aleksey Savrasov;a painting in the style of Adolph Gottlieb;a painting in the style of Adrianus Eversen;a painting in the style of Affandi;a painting in the style of Alessandro Allori;a painting in the style of Alec Soth;a painting in the style of Abigail Larson;a painting in the style of Alan Parry;a painting in the style of Akira Toriyama;a painting in the style of Adonna Khare;a painting in the style of Alena Aenami;a painting in the style of Agnes Cecile;a painting in the style of Albert Lynch;a painting in the style of Albrecht Durer;a painting in the style of Alejandro Jodorowsky;a painting in the style of Alex Alemany;a painting in the style of Alex Colville;a painting in the style of Albert Eckhout;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adolph Menzel;a painting in the style of Aleksi Briclot;a painting in the style of Agostino Tassi;a painting in the style of Albert Bierstadt;a painting in the style of Aaron Douglas;a painting in the style of Albert Tucker;a painting in the style of Abraham Pether;a painting in the style of Alberto Magnelli;a painting in the style of Adam Hughes;a painting in the style of Alan Schaller" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG0.50_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG1.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG2.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.nG3.00_U.picasso_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*cocoval.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 

  accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="*cocoval.5000" --instance_prompt="*coco30k.5000" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --donot_reinit_validation_generator \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 



# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.20P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
# echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
# '
#                             accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.20P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*coco30k.500" --instance_prompt="*coco30k.500" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.80P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.80P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Octavia Spencer;a photo of Anne Hathaway;a photo of Idris Elba;a photo of Morgan Freeman;a photo of Oprah Winfrey;a photo of Emma Stone;a photo of Elon Musk" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P16.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P64.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.40Ir0.60P128.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 50 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4 1000
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
                                --load_unet_weight_path="data_root/logs/esd/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000.safetensors" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/esd-x-kv.bG.fG.T999-1000.peUG-PS1.00_1.00AtE0.20Ir0.60P32.00-N0.00G0.00-mt_U.adriver_sd1.4.bf16.bs4_r0/step1000" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" --instance_prompt="a photo of Adam Driver;a photo of Andrew Garfield;a photo of Anne Hathaway;a photo of Amy Adams;a photo of Emma Stone;a photo of Amber Heard;a photo of Morgan Freeman;a photo of Chris Rock;a photo of Will Smith;a photo of Idris Elba;a photo of Oprah Winfrey;a photo of Elon Musk;a photo of Zendaya;a photo of Nicki Minaj;a photo of Octavia Spencer" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 100 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+4_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+6_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb5e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb2.5e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
echo 'count:0 - mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16 0
'
                            accelerate launch train_dreambooth_lora.py \
                                --pretrained_model_name_or_path='data_root2/logs/mace/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/LoRA_fusion_model'  \
                                --load_unet_weight_path="" \
                                --load_lora_weight_path="" \
                                --instance_data_dir="data_root/data/real_data/dummy" \
                                --gen_image_path="data_root/generated/study/mace.psg8e-1.tr1e0.fr1e0.lamb1e+5_U.picasso_sd1.4.bf16_r0/step0" \
                                --output_dir="data_root/logs/gen" \
                                --validation_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" --instance_prompt="a painting in the style of Picasso;a painting in the style of Van Gogh;a painting in the style of Claude Monet;a painting in the style of Jackson Pollock;a painting in the style of A.J. Casson;a painting in the style of Aaron Douglas;a painting in the style of Aaron Horkey;a painting in the style of Aaron Jasinski;a painting in the style of Aaron Siskind;a painting in the style of Abbott Fuller Graves;a painting in the style of Abbott Handerson Thayer;a painting in the style of Abdel Hadi Al Gazzar;a painting in the style of Abed Abdi;a painting in the style of Abigail Larson;a painting in the style of Abraham Mintchine;a painting in the style of Abraham Pether;a painting in the style of Abram Efimovich Arkhipov;a painting in the style of Adam Elsheimer;a painting in the style of Adam Hughes;a painting in the style of Adam Martinakis;a painting in the style of Adam Paquette;a painting in the style of Adi Granov;a painting in the style of Adolf Hirémy-Hirschl;a painting in the style of Adolph Gottlieb;a painting in the style of Adolph Menzel;a painting in the style of Adonna Khare;a painting in the style of Adriaen van Ostade;a painting in the style of Adriaen van Outrecht;a painting in the style of Adrian Donoghue;a painting in the style of Adrian Ghenie;a painting in the style of Adrian Paul Allinson;a painting in the style of Adrian Smith;a painting in the style of Adrian Tomine;a painting in the style of Adrianus Eversen;a painting in the style of Afarin Sajedi;a painting in the style of Affandi;a painting in the style of Aggi Erguna;a painting in the style of Agnes Cecile;a painting in the style of Agnes Lawrence Pelton;a painting in the style of Agnes Martin;a painting in the style of Agostino Arrivabene;a painting in the style of Agostino Tassi;a painting in the style of Ai Weiwei;a painting in the style of Ai Yazawa;a painting in the style of Akihiko Yoshida;a painting in the style of Akira Toriyama;a painting in the style of Akos Major;a painting in the style of Akseli Gallen-Kallela;a painting in the style of Al Capp;a painting in the style of Al Feldstein;a painting in the style of Al Williamson;a painting in the style of Alain Laboile;a painting in the style of Alan Bean;a painting in the style of Alan Davis;a painting in the style of Alan Kenny;a painting in the style of Alan Lee;a painting in the style of Alan Moore;a painting in the style of Alan Parry;a painting in the style of Alan Schaller;a painting in the style of Alasdair McLellan;a painting in the style of Alastair Magnaldo;a painting in the style of Alayna Lemmer;a painting in the style of Albert Benois;a painting in the style of Albert Bierstadt;a painting in the style of Albert Bloch;a painting in the style of Albert Dubois-Pillet;a painting in the style of Albert Eckhout;a painting in the style of Albert Edelfelt;a painting in the style of Albert Gleizes;a painting in the style of Albert Goodwin;a painting in the style of Albert Joseph Moore;a painting in the style of Albert Koetsier;a painting in the style of Albert Kotin;a painting in the style of Albert Lynch;a painting in the style of Albert Marquet;a painting in the style of Albert Pinkham Ryder;a painting in the style of Albert Robida;a painting in the style of Albert Servaes;a painting in the style of Albert Tucker;a painting in the style of Albert Watson;a painting in the style of Alberto Biasi;a painting in the style of Alberto Burri;a painting in the style of Alberto Giacometti;a painting in the style of Alberto Magnelli;a painting in the style of Alberto Seveso;a painting in the style of Alberto Sughi;a painting in the style of Alberto Vargas;a painting in the style of Albrecht Anker;a painting in the style of Albrecht Durer;a painting in the style of Alec Soth;a painting in the style of Alejandro Burdisio;a painting in the style of Alejandro Jodorowsky;a painting in the style of Aleksey Savrasov;a painting in the style of Aleksi Briclot;a painting in the style of Alena Aenami;a painting in the style of Alessandro Allori;a painting in the style of Alessandro Barbucci;a painting in the style of Alessandro Gottardo;a painting in the style of Alessio Albi;a painting in the style of Alex Alemany;a painting in the style of Alex Andreev;a painting in the style of Alex Colville;a painting in the style of Alex Figini;a painting in the style of Alex Garant" \
                                --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
                                --run_note 'gen img' --wait_weight \
                                --num_validation_images 25 \
                                --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 


#   accelerate launch train_dreambooth_lora.py \
#                                 --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'  \
#                                 --load_unet_weight_path="" \
#                                 --load_lora_weight_path="" \
#                                 --instance_data_dir="data_root/data/real_data/dummy" \
#                                 --gen_image_path="data_root/generated/study/original_pretrained_sd1.4_bf16" \
#                                 --output_dir="data_root/logs/gen" \
#                                 --validation_prompt="*cocoval.5000" --instance_prompt="*coco30k.5000" \
#                                 --lora_rank 1 --target_lora_modules to_k to_v --target_lora_layers cross \
#                                 --run_note 'gen img' --wait_weight \
#                                 --num_validation_images 100 \
#                                 --donot_reinit_validation_generator \
#                                 --cfg_scale 7.50 --gen_batch 10 --gen_dtype "bf16" 
