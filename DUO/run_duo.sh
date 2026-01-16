export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3



        accelerate launch               \
            --main_process_port 50000   \
            unlearn-sd_custom.py               \
            --project="SD-DPO_survival-no_prompt"               \
            --mixed_precision="bf16"          \
            --group=""                                          \
            --config_dir="datasets/SD/config.json"    \
            --config_name="naked_person"                      \
            --data_dir="../data_root/generated/duo"                  \
            --output_dir="../data_root/logs/duo/duo-s_U.naked_sd1.4.bf16_r0"                          \
            --target_prompt="naked person"                     \
            --synonym_prompt=""                      \
            --prior_prompt="dressed person"                              \
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
            --no_grad=""                                        \
            --train_method="duo-s"                          \
            --seed=42
        
['duo-s_U.naked_sd1.4.bf16_r0']


#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="Margot_Robbie"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="Margot Robbie"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="person"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        

#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="David_Beckham"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="David Beckham"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="person"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        

#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="Rihanna"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="Rihanna"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="person"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        

#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="Barack_Obama"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.obama_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="Barack Obama"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="person"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        
# ['duo-s_U.mrobbie_sd1.4.bf16.bs4_r0', 'duo-s_U.beckham_sd1.4.bf16.bs4_r0', 'duo-s_U.rihanna_sd1.4.bf16.bs4_r0', 'duo-s_U.obama_sd1.4.bf16.bs4_r0']







#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="a_painting_in_the_style_of_Picasso"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="a painting in the style of Picasso"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="a painting in the style of artist"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        

#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="a painting in the style of Jackson Pollock"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="a painting in the style of artist"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        

#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="a painting in the style of Van Gogh"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="a painting in the style of artist"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        

#         accelerate launch               \
#             --main_process_port 50000   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="datasets/SD/config.json"    \
#             --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16.bs4_r0"                          \
#             --target_prompt="a painting in the style of Claude Monet"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="a painting in the style of artist"                              \
#             --base_lr=3e-4                                      \
#             --adam_weight_decay=1e-2                            \
#             --dcoloss_beta=500                       \
#             --base_lambda=1e6                                   \
#             --rank=32                                           \
#             --method=dpo                                        \
#             --train_batch_size=1                                \
#             --max_train_steps=1000                              \
#             --checkpointing_steps=250                           \
#             --validation_steps=250                              \
#             --num_validation_images=2                           \
#             --num_samples=64                         \
#             --t_max=750                                         \
#             --t_min=1                                           \
#             --no_grad=""                                        \
#             --train_method="duo-s"                          \
#             --seed=42
        
# ['duo-s_U.picasso_sd1.4.bf16.bs4_r0', 'duo-s_U.pollock_sd1.4.bf16.bs4_r0', 'duo-s_U.vgogh_sd1.4.bf16.bs4_r0', 'duo-s_U.cmonet_sd1.4.bf16.bs4_r0']


