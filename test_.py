

concept = 'chiquita' # moodeng
data_setting = 'full' # full
is_relearn = False # True

lora_rank = 4 # 1
use_pr = True
use_ti = True # True 
use_ni = False

lr_lora = "2.5e-4" # 1e-4
lr_ti = "1e-2" #  apply only if use_ti



if data_setting == 'fewshot':
    
    if concept == 'avp':
        dataset_name = 'avpS3'
    else:
        dataset_name = f'{concept}U3'
else:
    if concept == 'avp':
        dataset_name = 'avp20'
    else:
        dataset_name = f'{concept}50'

dataset_name2data_root = {
    'crybabyU3': 'data_root/data/real_data/crybaby/crybaby-unseen-3',
    'crybaby50': 'data_root/data/real_data/crybaby/crybaby-50',
    'moodengU3': 'data_root/data/real_data/moodeng/moodeng-unseen-3',
    'moodeng50': 'data_root/data/real_data/moodeng/moodeng-50',
    'chiquita50': 'data_root/data/real_data/chiquita/chiquita-50',
    'chiquitaU3': 'data_root/data/real_data/chiquita/chiquita-unseen-3',
    'avp20': 'data_root/data/real_data/avp/avp-20',
    'avpS3': 'data_root/data/real_data/avp/avp-seen-3',
}
concept2prompt = {
    'crybaby': 'A photo of a crybaby art toy',
    'moodeng': 'A photo of a cute baby hippo',
}
concept2generalprompt = {
    'crybaby': 'A photo of a toy',
    'moodeng': 'A photo of a hippo',
    
}
concept2initializer = {
    'crybaby': 'toy',
    'moodeng': 'hippo',
    'chiquita': 'girl', 
    'avp': 'glasses',
}

concept2Prprompt = {
    'crybaby': 'A photo of a toy',
    'moodeng': 'A photo of a hippo',
    'chiquita': 'A photo of a girl',
    'avp': 'A photo of a glasses',
}
# however, if use_ti is True, the prompt will be changed to 'A photo of a v1' for all concepts
data_root = dataset_name2data_root[dataset_name]
if use_ti:
    prompt = 'A photo of a v1' 
else:
    prompt = concept2prompt[concept]
    
pretrained_path = 'CompVis/stable-diffusion-v1-4' 
if  is_relearn:
    pretrained_path = f"data_root/logs/erase_l1.{concept}VPr.object_lr2.5e-4/LoRA_fusion_model"

        
if use_ti:
    dataset_name_for_exp = dataset_name + "-V"
    if use_ni:
        dataset_name_for_exp += ".ni"
else: dataset_name_for_exp = dataset_name

exp_name = f'c.l{lora_rank}.kv_{dataset_name_for_exp}'
if use_pr:
    exp_name += f'_pr0.50'
exp_name += '_lr'
if lora_rank >0: exp_name += f"{str(lr_lora)}"
if use_ti:
    exp_name += f'.ti{str(lr_ti)}'
exp_name += '_f0.5_b1g4'
if is_relearn:
    unlearn_setting = pretrained_path.split("/")[-2].split("_")[1]
    exp_name = f'uul.{unlearn_setting}_{exp_name}'
    
if use_ni: initializer_token = ''
else: 
    initializer_token = concept2initializer[concept]



name_tag = ''
if is_relearn: name_tag += 'uul'
name_tag = f'{name_tag} {dataset_name}'
name_tag += f' l{lora_rank}'
if use_ti: 
    # name_tag += f' ti.{lr_ti}'
    name_tag += f' ti'


script = f"""
accelerate launch train_dreambooth_lora.py \\
  --pretrained_model_name_or_path={pretrained_path}  \\
  --instance_data_dir={data_root} \\
  --output_dir="data_root/logs/{exp_name}" \\
  --validation_prompt="{prompt}" --instance_prompt="{prompt}" \\
  --train_batch_size=1 --gradient_accumulation_steps=4 \\
  --lora_rank {lora_rank} --target_lora_modules to_k to_v --target_lora_layers cross \\
  --max_train_steps=3000  --validation_steps=250  --checkpointing_steps=50 \\
  --run_note '{name_tag}' \\"""
    
    
if use_pr:
    script += f"""
  --with_prior_preservation --prior_loss_weight=0.5 --num_class_images 50 \\
  --class_prompt="{concept2Prprompt[concept]}" --class_data_dir="data_root/generated/model/original_pretrained/{concept2Prprompt[concept]}/7.50" \\"""
    
# Conditional learning rate + TI options
if use_ti:
    
    if lora_rank <= 0:
        script += f"""
  --learning_rate_ti {lr_ti} \\
  --placeholder_token="v1" --initializer_token='{initializer_token}'"""
    else:
        script += f"""
  --learning_rate_lora {lr_lora} --learning_rate_ti {lr_ti} \\
  --placeholder_token="v1" --initializer_token='{initializer_token}'"""
else:
    script += f"""
  --learning_rate {lr_lora}"""

print(script)


