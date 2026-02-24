erase_concept="David Beckham"
generic_prompt="person"
train_method='xattn-kv'
train_data_dir="data/images/train/David_Beckham/"
prompt_template="person"
final_weight_path="stereo_weights_mine/David_Beckham/"
n_iterations=3
iterations=200
ti_max_train_steps=500
early_step=50
suma_step=3000


python -W ignore train.py --erase_concept $erase_concept --generic_prompt $generic_prompt --train_method $train_method --train_data_dir $train_data_dir --learnable_property $prompt_template --initializer_token 'toy' --output_dir $final_weight_path --mode stereo --unet_ckpt_to_attack final_reo_unet.pt --attack_eval_images data/images/eval/David_Beckham/ --mode stereo --compositional_guidance_scale 2 --n_iterations $n_iterations --iterations $iterations --ti_max_train_steps $ti_max_train_steps --num_of_adv_concepts 2 --anchor_concept_path utils/anchor_prompts.json --early_step $early_step
python train_suma.py --output_dir $final_weight_path --train_method $train_method --suma_step $suma_step --train_data_dir $train_data_dir --learnable_property $prompt_template

