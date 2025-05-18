import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from omegaconf import OmegaConf
import torch
from src.cfr_lora_training import main as cfr_lora_training
from src.fuse_lora_close_form import main as multi_lora_fusion
from inference import main as inference


def main(conf):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    

    # stage 1 & 2 (CFR and LoRA training)
    cfr_lora_training(conf.MACE)

    # stage 3 (Multi-LoRA fusion)
    multi_lora_fusion(conf.MACE) # fuse Multiple-LoRA with the original pretrained projection

    # test the erased model
    if conf.MACE.test_erased_model:
        inference(OmegaConf.create({
            "pretrained_model_name_or_path": conf.MACE.final_save_path,
            "multi_concept": conf.MACE.multi_concept,
            "generate_training_data": False,
            "device": device,
            "steps": 50,
            "output_dir": conf.MACE.final_save_path,
        }))


# if __name__ == "__main__":
#     conf_path = sys.argv[1]
#     conf = OmegaConf.load(conf_path)
#     main(conf)
# if __name__ == "__main__":
#     cli_conf = OmegaConf.from_cli()
#     yaml_conf = OmegaConf.load(cli_conf.pop("_args_")[0])  # first arg is YAML path
#     conf = OmegaConf.merge(yaml_conf, cli_conf)
#     main(conf)




if __name__ == "__main__":
    # First argument is the config path
    config_path = sys.argv[1]
    
    # Remaining arguments are overrides
    cli_overrides = sys.argv[2:]

    # Load base config
    yaml_conf = OmegaConf.load(config_path)

    # Load CLI overrides
    override_conf = OmegaConf.from_dotlist(cli_overrides)

    # Merge them
    conf = OmegaConf.merge(yaml_conf, override_conf)



    if "exp_name" in conf and conf.exp_name:
        print('overwrite output_dir by exp_name')
        exp = conf.exp_name
        conf.MACE.output_dir = f"./data_root/logs/{exp}/CFR_with_multi_LoRAs"
        conf.MACE.final_save_path = f"./data_root/logs/{exp}/LoRA_fusion_model"

        print(f"output_dir: {conf.MACE.output_dir}")
        print(f"final_save_path: {conf.MACE.final_save_path}")
        
    
    main(conf)
