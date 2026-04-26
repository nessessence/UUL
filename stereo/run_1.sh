# export CUDA_VISIBLE_DEVICES=1
export device="cuda:1"

# python generate_images.py  --prompt "a photo of Morgan Freeman" --output_dir "../data_root/generated/stereo/a photo of Morgan Freeman/"  --num_images 500 
# python generate_images.py  --prompt "a photo of Oprah Winfrey" --output_dir "../data_root/generated/stereo/a photo of Oprah Winfrey/"  --num_images 500 


python generate_images.py  --prompt "a painting in the style of Akira Toriyama" --output_dir "../data_root/generated/stereo/a painting in the style of Akira Toriyama/"  --num_images 500 
python generate_images.py  --prompt "a painting in the style of Georges Seurat" --output_dir "../data_root/generated/stereo/a painting in the style of Georges Seurat/"  --num_images 500 


# python -W ignore train.py --erase_concept 'a painting in the style of Van Gogh' --train_method noxattn --train_data_dir "../data_root/generated/stereo/a painting in the style of Van Gogh/" --learnable_property 'style' --initializer_token 'art' --output_dir "../data_root/logs/stereo/stereo_U.vgogh_sd1.4.bf16_r0" --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images  "../data_root/generated/stereo/A photo of a painting in the style of Van Gogh/" --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2   --anchor_concept_path utils/anchor_prompts_custom.json --seed 42 --device $device 
# python -W ignore train.py --erase_concept 'a painting in the style of Picasso' --train_method noxattn --train_data_dir "../data_root/generated/stereo/a painting in the style of Picasso/" --learnable_property 'style' --initializer_token 'art' --output_dir "../data_root/logs/stereo/stereo_U.picasso_sd1.4.bf16_r0" --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images  "../data_root/generated/stereo/A photo of a painting in the style of Picasso/" --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2   --anchor_concept_path utils/anchor_prompts_custom.json --seed 42 --device $device 
# python -W ignore train.py --erase_concept 'a painting in the style of Claude Monet' --train_method noxattn --train_data_dir "../data_root/generated/stereo/a painting in the style of Claude Monet/" --learnable_property 'style' --initializer_token 'art' --output_dir "../data_root/logs/stereo/stereo_U.cmonet_sd1.4.bf16_r0" --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images  "../data_root/generated/stereo/A photo of a painting in the style of Claude Monet/" --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2   --anchor_concept_path utils/anchor_prompts_custom.json --seed 42 --device $device 

# python generate_images.py  --prompt "A photo of Margot Robbie" --output_dir "../data_root/generated/stereo/A photo of Margot Robbie/"  --num_images 500 
# python generate_images.py  --prompt "A photo of mickey mouse" --output_dir "../data_root/generated/stereo/A photo of mickey mouse/"  --num_images 500 




# python generate_images.py  --prompt "A photo of Amy Adams" --output_dir "../data_root/generated/stereo/A photo of Amy Adams/"  --num_images 500 
# python -W ignore train.py --erase_concept 'Amy Adams' --train_method noxattn --train_data_dir "../data_root/generated/stereo/A photo of Amy Adams/" --learnable_property 'object' --initializer_token 'person' --output_dir "../data_root/logs/stereo/Amy Adams" --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images  "../data_root/generated/stereo/A photo of Amy Adams/" --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2   --anchor_concept_path utils/person_anchor_prompts.json