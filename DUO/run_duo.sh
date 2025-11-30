export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3



    accelerate launch               \
        --main_process_port 50000   \
        unlearn-sd_custom.py               \
        --project="SD-DPO_survival-no_prompt"               \
        --group=""                                          \
        --config_dir="datasets/SD/config.json"    \
        --config_name="Margot_Robbie"                      \
        --data_dir="../data_root/generated/duo"                  \
        --output_dir="../data_root/logs/study/duo/duo_mrobbie"                          \
        --target_prompt="Margot Robbie"                     \
        --synonym_prompt=""                      \
        --prior_prompt="person"                              \
        --base_lr=3e-4                                      \
        --adam_weight_decay=1e-2                            \
        --dcoloss_beta=500                       \
        --base_lambda=1e6                                   \
        --rank=32                                           \
        --method=dpo                                        \
        --train_batch_size=1                                \
        --max_train_steps=1000                              \
        --checkpointing_steps=250                           \
        --validation_steps=250                              \
        --num_validation_images=2                           \
        --num_samples=64                         \
        --t_max=750                                         \
        --t_min=1                                           \
        --no_grad ""                                        \
        --no_cross_attn                                     \
        --seed=42
    

    accelerate launch               \
        --main_process_port 50000   \
        unlearn-sd_custom.py               \
        --project="SD-DPO_survival-no_prompt"               \
        --group=""                                          \
        --config_dir="datasets/SD/config.json"    \
        --config_name="Donald_Trump"                      \
        --data_dir="../data_root/generated/duo"                  \
        --output_dir="../data_root/logs/study/duo/duo_dtrump"                          \
        --target_prompt="Donald Trump"                     \
        --synonym_prompt=""                      \
        --prior_prompt="person"                              \
        --base_lr=3e-4                                      \
        --adam_weight_decay=1e-2                            \
        --dcoloss_beta=500                       \
        --base_lambda=1e6                                   \
        --rank=32                                           \
        --method=dpo                                        \
        --train_batch_size=1                                \
        --max_train_steps=1000                              \
        --checkpointing_steps=250                           \
        --validation_steps=250                              \
        --num_validation_images=2                           \
        --num_samples=64                         \
        --t_max=750                                         \
        --t_min=1                                           \
        --no_grad ""                                        \
        --no_cross_attn                                     \
        --seed=42
    

    accelerate launch               \
        --main_process_port 50000   \
        unlearn-sd_custom.py               \
        --project="SD-DPO_survival-no_prompt"               \
        --group=""                                          \
        --config_dir="datasets/SD/config.json"    \
        --config_name="Barack_Obama"                      \
        --data_dir="../data_root/generated/duo"                  \
        --output_dir="../data_root/logs/study/duo/duo_obama"                          \
        --target_prompt="Barack Obama"                     \
        --synonym_prompt=""                      \
        --prior_prompt="person"                              \
        --base_lr=3e-4                                      \
        --adam_weight_decay=1e-2                            \
        --dcoloss_beta=500                       \
        --base_lambda=1e6                                   \
        --rank=32                                           \
        --method=dpo                                        \
        --train_batch_size=1                                \
        --max_train_steps=1000                              \
        --checkpointing_steps=250                           \
        --validation_steps=250                              \
        --num_validation_images=2                           \
        --num_samples=64                         \
        --t_max=750                                         \
        --t_min=1                                           \
        --no_grad ""                                        \
        --no_cross_attn                                     \
        --seed=42
    

    accelerate launch               \
        --main_process_port 50000   \
        unlearn-sd_custom.py               \
        --project="SD-DPO_survival-no_prompt"               \
        --group=""                                          \
        --config_dir="datasets/SD/config.json"    \
        --config_name="mickey_mouse"                      \
        --data_dir="../data_root/generated/duo"                  \
        --output_dir="../data_root/logs/duo/duo_mmouse"                          \
        --target_prompt="mickey mouse"                     \
        --synonym_prompt=""                      \
        --prior_prompt="cartoon character"                              \
        --base_lr=3e-4                                      \
        --adam_weight_decay=1e-2                            \
        --dcoloss_beta=500                       \
        --base_lambda=1e6                                   \
        --rank=32                                           \
        --method=dpo                                        \
        --train_batch_size=1                                \
        --max_train_steps=1000                              \
        --checkpointing_steps=250                           \
        --validation_steps=250                              \
        --num_validation_images=2                           \
        --num_samples=64                         \
        --t_max=750                                         \
        --t_min=1                                           \
        --no_grad ""                                        \
        --no_cross_attn                                     \
        --seed=42
    