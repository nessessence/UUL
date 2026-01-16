export device="cuda:2"


base_dir=$(pwd)
cd $base_dir/stereo

python generate_images.py  --prompt "a photo of naked person" --output_dir "../data_root/generated/stereo/a photo of naked person/"  --num_images 500 
python -W ignore train.py --erase_concept 'naked person' --train_method noxattn --train_data_dir "../data_root/generated/stereo/a photo of naked person/" --learnable_property 'object' --initializer_token 'person' --output_dir "../data_root/logs/stereo/stereo_U.naked_sd1.4.bf16_r0" --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images  "../data_root/generated/stereo/a photo of naked person/" --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2   --anchor_concept_path utils/anchor_prompts_custom.json --seed 42 --device $device 