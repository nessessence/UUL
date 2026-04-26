export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3


            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Morgan_Freeman"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.mfreeman_sd1.4.bf16_r0"                          \
                --target_prompt="Morgan Freeman"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Morgan_Freeman"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.mfreeman_sd1.4.bf16_r0"                          \
                --target_prompt="Morgan Freeman"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Morgan_Freeman"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.mfreeman_sd1.4.bf16_r0"                          \
                --target_prompt="Morgan Freeman"                     \
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
                --no_grad=""                                        \
                --train_method="duo-s"                          \
                --seed=42
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Morgan_Freeman"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.mfreeman_sd1.4.bf16_r0"                          \
                --target_prompt="Morgan Freeman"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Oprah_Winfrey"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.owinfrey_sd1.4.bf16_r0"                          \
                --target_prompt="Oprah Winfrey"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Oprah_Winfrey"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.owinfrey_sd1.4.bf16_r0"                          \
                --target_prompt="Oprah Winfrey"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Oprah_Winfrey"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.owinfrey_sd1.4.bf16_r0"                          \
                --target_prompt="Oprah Winfrey"                     \
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
                --no_grad=""                                        \
                --train_method="duo-s"                          \
                --seed=42
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Oprah_Winfrey"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.owinfrey_sd1.4.bf16_r0"                          \
                --target_prompt="Oprah Winfrey"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Emma_Stone"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.estone_sd1.4.bf16_r0"                          \
                --target_prompt="Emma Stone"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Emma_Stone"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.estone_sd1.4.bf16_r0"                          \
                --target_prompt="Emma Stone"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Emma_Stone"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.estone_sd1.4.bf16_r0"                          \
                --target_prompt="Emma Stone"                     \
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
                --no_grad=""                                        \
                --train_method="duo-s"                          \
                --seed=42
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Emma_Stone"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.estone_sd1.4.bf16_r0"                          \
                --target_prompt="Emma Stone"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Elon_Musk"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.elon_sd1.4.bf16_r0"                          \
                --target_prompt="Elon Musk"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Elon_Musk"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.elon_sd1.4.bf16_r0"                          \
                --target_prompt="Elon Musk"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Elon_Musk"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.elon_sd1.4.bf16_r0"                          \
                --target_prompt="Elon Musk"                     \
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
                --no_grad=""                                        \
                --train_method="duo-s"                          \
                --seed=42
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="Elon_Musk"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.elon_sd1.4.bf16_r0"                          \
                --target_prompt="Elon Musk"                     \
                --synonym_prompt=""                      \
                --prior_prompt="person"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            
['duo-s.b100_U.mfreeman_sd1.4.bf16_r0', 'duo-s.b250_U.mfreeman_sd1.4.bf16_r0', 'duo-s.b500_U.mfreeman_sd1.4.bf16_r0', 'duo-s.b1000_U.mfreeman_sd1.4.bf16_r0', 'duo-s.b100_U.owinfrey_sd1.4.bf16_r0', 'duo-s.b250_U.owinfrey_sd1.4.bf16_r0', 'duo-s.b500_U.owinfrey_sd1.4.bf16_r0', 'duo-s.b1000_U.owinfrey_sd1.4.bf16_r0', 'duo-s.b100_U.estone_sd1.4.bf16_r0', 'duo-s.b250_U.estone_sd1.4.bf16_r0', 'duo-s.b500_U.estone_sd1.4.bf16_r0', 'duo-s.b1000_U.estone_sd1.4.bf16_r0', 'duo-s.b100_U.elon_sd1.4.bf16_r0', 'duo-s.b250_U.elon_sd1.4.bf16_r0', 'duo-s.b500_U.elon_sd1.4.bf16_r0', 'duo-s.b1000_U.elon_sd1.4.bf16_r0']



            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Akira_Toriyama"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.toriyama_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Akira Toriyama"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Akira_Toriyama"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.toriyama_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Akira Toriyama"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Akira_Toriyama"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.toriyama_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Akira Toriyama"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Akira_Toriyama"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.toriyama_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Akira Toriyama"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Georges_Seurat"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.gsrat_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Georges Seurat"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Georges_Seurat"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.gsrat_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Georges Seurat"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Georges_Seurat"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.gsrat_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Georges Seurat"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Georges_Seurat"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.gsrat_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Georges Seurat"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Joan_Miro"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.jmiro_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Joan Miro"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Joan_Miro"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.jmiro_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Joan Miro"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Joan_Miro"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.jmiro_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Joan Miro"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Joan_Miro"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.jmiro_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Joan Miro"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Egon_Schiele"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b100_U.egon_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Egon Schiele"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=100                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Egon_Schiele"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b250_U.egon_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Egon Schiele"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=250                       \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Egon_Schiele"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b500_U.egon_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Egon Schiele"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
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
            

            accelerate launch               \
                --main_process_port 50000   \
                unlearn-sd_custom.py               \
                --project="SD-DPO_survival-no_prompt"               \
                --mixed_precision="bf16"          \
                --group=""                                          \
                --config_dir="../data_root/generated/duo/config.json"    \
                --config_name="a_painting_in_the_style_of_Egon_Schiele"                      \
                --data_dir="../data_root/generated/duo"                  \
                --output_dir="../data_root/logs/duo/duo-s.b1000_U.egon_sd1.4.bf16_r0"                          \
                --target_prompt="a painting in the style of Egon Schiele"                     \
                --synonym_prompt=""                      \
                --prior_prompt="a painting in the style of artist"                              \
                --base_lr=3e-4                                      \
                --adam_weight_decay=1e-2                            \
                --dcoloss_beta=1000                       \
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
            
['duo-s.b100_U.toriyama_sd1.4.bf16_r0', 'duo-s.b250_U.toriyama_sd1.4.bf16_r0', 'duo-s.b500_U.toriyama_sd1.4.bf16_r0', 'duo-s.b1000_U.toriyama_sd1.4.bf16_r0', 'duo-s.b100_U.gsrat_sd1.4.bf16_r0', 'duo-s.b250_U.gsrat_sd1.4.bf16_r0', 'duo-s.b500_U.gsrat_sd1.4.bf16_r0', 'duo-s.b1000_U.gsrat_sd1.4.bf16_r0', 'duo-s.b100_U.jmiro_sd1.4.bf16_r0', 'duo-s.b250_U.jmiro_sd1.4.bf16_r0', 'duo-s.b500_U.jmiro_sd1.4.bf16_r0', 'duo-s.b1000_U.jmiro_sd1.4.bf16_r0', 'duo-s.b100_U.egon_sd1.4.bf16_r0', 'duo-s.b250_U.egon_sd1.4.bf16_r0', 'duo-s.b500_U.egon_sd1.4.bf16_r0', 'duo-s.b1000_U.egon_sd1.4.bf16_r0']


#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            
# ['duo-s.b100_U.naked_sd1.4.bf16_r0', 'duo-s.b250_U.naked_sd1.4.bf16_r0', 'duo-s.b500_U.naked_sd1.4.bf16_r0', 'duo-s.b1000_U.naked_sd1.4.bf16_r0']

# sleep 3h

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_woman"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.nakedw_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked woman"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed woman"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_woman"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.nakedw_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked woman"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed woman"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_woman"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.nakedw_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked woman"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed woman"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_woman"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.nakedw_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked woman"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed woman"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b100_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b250_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b500_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 50000   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-s.b1000_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-s"                          \
#                 --seed=42
            
# ['duo-s.b100_U.mrobbie_sd1.4.bf16_r0', 'duo-s.b250_U.mrobbie_sd1.4.bf16_r0', 'duo-s.b500_U.mrobbie_sd1.4.bf16_r0', 'duo-s.b1000_U.mrobbie_sd1.4.bf16_r0', 'duo-s.b100_U.beckham_sd1.4.bf16_r0', 'duo-s.b250_U.beckham_sd1.4.bf16_r0', 'duo-s.b500_U.beckham_sd1.4.bf16_r0', 'duo-s.b1000_U.beckham_sd1.4.bf16_r0', 'duo-s.b100_U.rihanna_sd1.4.bf16_r0', 'duo-s.b250_U.rihanna_sd1.4.bf16_r0', 'duo-s.b500_U.rihanna_sd1.4.bf16_r0', 'duo-s.b1000_U.rihanna_sd1.4.bf16_r0', 'duo-s.b100_U.obama_sd1.4.bf16_r0', 'duo-s.b250_U.obama_sd1.4.bf16_r0', 'duo-s.b500_U.obama_sd1.4.bf16_r0', 'duo-s.b1000_U.obama_sd1.4.bf16_r0', 'duo-s.b100_U.picasso_sd1.4.bf16_r0', 'duo-s.b250_U.picasso_sd1.4.bf16_r0', 'duo-s.b500_U.picasso_sd1.4.bf16_r0', 'duo-s.b1000_U.picasso_sd1.4.bf16_r0', 'duo-s.b100_U.pollock_sd1.4.bf16_r0', 'duo-s.b250_U.pollock_sd1.4.bf16_r0', 'duo-s.b500_U.pollock_sd1.4.bf16_r0', 'duo-s.b1000_U.pollock_sd1.4.bf16_r0', 'duo-s.b100_U.vgogh_sd1.4.bf16_r0', 'duo-s.b250_U.vgogh_sd1.4.bf16_r0', 'duo-s.b500_U.vgogh_sd1.4.bf16_r0', 'duo-s.b1000_U.vgogh_sd1.4.bf16_r0', 'duo-s.b100_U.cmonet_sd1.4.bf16_r0', 'duo-s.b250_U.cmonet_sd1.4.bf16_r0', 'duo-s.b500_U.cmonet_sd1.4.bf16_r0', 'duo-s.b1000_U.cmonet_sd1.4.bf16_r0', 'duo-s.b100_U.nakedw_sd1.4.bf16_r0', 'duo-s.b250_U.nakedw_sd1.4.bf16_r0', 'duo-s.b500_U.nakedw_sd1.4.bf16_r0', 'duo-s.b1000_U.nakedw_sd1.4.bf16_r0', 'duo-s.b100_U.nakedm_sd1.4.bf16_r0', 'duo-s.b250_U.nakedm_sd1.4.bf16_r0', 'duo-s.b500_U.nakedm_sd1.4.bf16_r0', 'duo-s.b1000_U.nakedm_sd1.4.bf16_r0', 'duo-s.b100_U.mmouse_sd1.4.bf16_r0', 'duo-s.b250_U.mmouse_sd1.4.bf16_r0', 'duo-s.b500_U.mmouse_sd1.4.bf16_r0', 'duo-s.b1000_U.mmouse_sd1.4.bf16_r0', 'duo-s.b100_U.r2d2_sd1.4.bf16_r0', 'duo-s.b250_U.r2d2_sd1.4.bf16_r0', 'duo-s.b500_U.r2d2_sd1.4.bf16_r0', 'duo-s.b1000_U.r2d2_sd1.4.bf16_r0', 'duo-s.b100_U.gcat_sd1.4.bf16_r0', 'duo-s.b250_U.gcat_sd1.4.bf16_r0', 'duo-s.b500_U.gcat_sd1.4.bf16_r0', 'duo-s.b1000_U.gcat_sd1.4.bf16_r0', 'duo-s.b100_U.macbook_sd1.4.bf16_r0', 'duo-s.b250_U.macbook_sd1.4.bf16_r0', 'duo-s.b500_U.macbook_sd1.4.bf16_r0', 'duo-s.b1000_U.macbook_sd1.4.bf16_r0']


#             # accelerate launch               \
#             #     --main_process_port 51234   \
#             #     unlearn-sd_custom.py               \
#             #     --project="SD-DPO_survival-no_prompt"               \
#             #     --mixed_precision="bf16"          \
#             #     --group=""                                          \
#             #     --config_dir="../data_root/generated/duo/config.json"    \
#             #     --config_name="naked_woman"                      \
#             #     --data_dir="../data_root/generated/duo"                  \
#             #     --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.nakedw_sd1.4.bf16_r0"                          \
#             #     --target_prompt="naked woman"                     \
#             #     --synonym_prompt=""                      \
#             #     --prior_prompt="dressed woman"                              \
#             #     --base_lr=3e-4                                      \
#             #     --adam_weight_decay=1e-2                            \
#             #     --dcoloss_beta=100                       \
#             #     --base_lambda=1e6                                   \
#             #     --rank=32                                           \
#             #     --method=dpo                                        \
#             #     --train_batch_size=1                                \
#             #     --max_train_steps=1000                              \
#             #     --checkpointing_steps=250                           \
#             #     --validation_steps=250                              \
#             #     --num_validation_images=2                           \
#             #     --num_samples=64                         \
#             #     --t_max=750                                         \
#             #     --t_min=1                                           \
#             #     --no_grad=""                                        \
#             #     --train_method="duo-x-kv"                          \
#             #     --seed=42
            

#             # accelerate launch               \
#             #     --main_process_port 51234   \
#             #     unlearn-sd_custom.py               \
#             #     --project="SD-DPO_survival-no_prompt"               \
#             #     --mixed_precision="bf16"          \
#             #     --group=""                                          \
#             #     --config_dir="../data_root/generated/duo/config.json"    \
#             #     --config_name="naked_woman"                      \
#             #     --data_dir="../data_root/generated/duo"                  \
#             #     --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.nakedw_sd1.4.bf16_r0"                          \
#             #     --target_prompt="naked woman"                     \
#             #     --synonym_prompt=""                      \
#             #     --prior_prompt="dressed woman"                              \
#             #     --base_lr=3e-4                                      \
#             #     --adam_weight_decay=1e-2                            \
#             #     --dcoloss_beta=250                       \
#             #     --base_lambda=1e6                                   \
#             #     --rank=32                                           \
#             #     --method=dpo                                        \
#             #     --train_batch_size=1                                \
#             #     --max_train_steps=1000                              \
#             #     --checkpointing_steps=250                           \
#             #     --validation_steps=250                              \
#             #     --num_validation_images=2                           \
#             #     --num_samples=64                         \
#             #     --t_max=750                                         \
#             #     --t_min=1                                           \
#             #     --no_grad=""                                        \
#             #     --train_method="duo-x-kv"                          \
#             #     --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_woman"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.nakedw_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked woman"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed woman"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_woman"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.nakedw_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked woman"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed woman"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_man"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.nakedm_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked man"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed man"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Mickey_Mouse"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.mmouse_sd1.4.bf16_r0"                          \
#                 --target_prompt="Mickey Mouse"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cartoon"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="R2D2_robot"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.r2d2_sd1.4.bf16_r0"                          \
#                 --target_prompt="R2D2 robot"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="robot"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Grumpy_Cat"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.gcat_sd1.4.bf16_r0"                          \
#                 --target_prompt="Grumpy Cat"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="cat"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Macbook"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.macbook_sd1.4.bf16_r0"                          \
#                 --target_prompt="Macbook"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="laptop"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            
# ['duo-x-kv.b100_U.nakedw_sd1.4.bf16_r0', 'duo-x-kv.b250_U.nakedw_sd1.4.bf16_r0', 'duo-x-kv.b500_U.nakedw_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.nakedw_sd1.4.bf16_r0', 'duo-x-kv.b100_U.nakedm_sd1.4.bf16_r0', 'duo-x-kv.b250_U.nakedm_sd1.4.bf16_r0', 'duo-x-kv.b500_U.nakedm_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.nakedm_sd1.4.bf16_r0', 'duo-x-kv.b100_U.mmouse_sd1.4.bf16_r0', 'duo-x-kv.b250_U.mmouse_sd1.4.bf16_r0', 'duo-x-kv.b500_U.mmouse_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.mmouse_sd1.4.bf16_r0', 'duo-x-kv.b100_U.r2d2_sd1.4.bf16_r0', 'duo-x-kv.b250_U.r2d2_sd1.4.bf16_r0', 'duo-x-kv.b500_U.r2d2_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.r2d2_sd1.4.bf16_r0', 'duo-x-kv.b100_U.gcat_sd1.4.bf16_r0', 'duo-x-kv.b250_U.gcat_sd1.4.bf16_r0', 'duo-x-kv.b500_U.gcat_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.gcat_sd1.4.bf16_r0', 'duo-x-kv.b100_U.macbook_sd1.4.bf16_r0', 'duo-x-kv.b250_U.macbook_sd1.4.bf16_r0', 'duo-x-kv.b500_U.macbook_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.macbook_sd1.4.bf16_r0']


#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Picasso"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.picasso_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Picasso"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.pollock_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Jackson Pollock"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.vgogh_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Van Gogh"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.cmonet_sd1.4.bf16_r0"                          \
#                 --target_prompt="a painting in the style of Claude Monet"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="a painting in the style of artist"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            
# ['duo-x-kv.b100_U.picasso_sd1.4.bf16_r0', 'duo-x-kv.b250_U.picasso_sd1.4.bf16_r0', 'duo-x-kv.b500_U.picasso_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.picasso_sd1.4.bf16_r0', 'duo-x-kv.b100_U.pollock_sd1.4.bf16_r0', 'duo-x-kv.b250_U.pollock_sd1.4.bf16_r0', 'duo-x-kv.b500_U.pollock_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.pollock_sd1.4.bf16_r0', 'duo-x-kv.b100_U.vgogh_sd1.4.bf16_r0', 'duo-x-kv.b250_U.vgogh_sd1.4.bf16_r0', 'duo-x-kv.b500_U.vgogh_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.vgogh_sd1.4.bf16_r0', 'duo-x-kv.b100_U.cmonet_sd1.4.bf16_r0', 'duo-x-kv.b250_U.cmonet_sd1.4.bf16_r0', 'duo-x-kv.b500_U.cmonet_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.cmonet_sd1.4.bf16_r0']



#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="naked_person"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.naked_sd1.4.bf16_r0"                          \
#                 --target_prompt="naked person"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="dressed person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            
# ['duo-x-kv.b100_U.naked_sd1.4.bf16_r0', 'duo-x-kv.b250_U.naked_sd1.4.bf16_r0', 'duo-x-kv.b500_U.naked_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.naked_sd1.4.bf16_r0']



#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Margot_Robbie"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.mrobbie_sd1.4.bf16_r0"                          \
#                 --target_prompt="Margot Robbie"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="David_Beckham"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.beckham_sd1.4.bf16_r0"                          \
#                 --target_prompt="David Beckham"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Rihanna"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.rihanna_sd1.4.bf16_r0"                          \
#                 --target_prompt="Rihanna"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b100_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=100                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b250_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=250                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b500_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=500                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            

#             accelerate launch               \
#                 --main_process_port 51234   \
#                 unlearn-sd_custom.py               \
#                 --project="SD-DPO_survival-no_prompt"               \
#                 --mixed_precision="bf16"          \
#                 --group=""                                          \
#                 --config_dir="../data_root/generated/duo/config.json"    \
#                 --config_name="Barack_Obama"                      \
#                 --data_dir="../data_root/generated/duo"                  \
#                 --output_dir="../data_root/logs/duo/duo-x-kv.b1000_U.obama_sd1.4.bf16_r0"                          \
#                 --target_prompt="Barack Obama"                     \
#                 --synonym_prompt=""                      \
#                 --prior_prompt="person"                              \
#                 --base_lr=3e-4                                      \
#                 --adam_weight_decay=1e-2                            \
#                 --dcoloss_beta=1000                       \
#                 --base_lambda=1e6                                   \
#                 --rank=32                                           \
#                 --method=dpo                                        \
#                 --train_batch_size=1                                \
#                 --max_train_steps=1000                              \
#                 --checkpointing_steps=250                           \
#                 --validation_steps=250                              \
#                 --num_validation_images=2                           \
#                 --num_samples=64                         \
#                 --t_max=750                                         \
#                 --t_min=1                                           \
#                 --no_grad=""                                        \
#                 --train_method="duo-x-kv"                          \
#                 --seed=42
            
# ['duo-x-kv.b100_U.mrobbie_sd1.4.bf16_r0', 'duo-x-kv.b250_U.mrobbie_sd1.4.bf16_r0', 'duo-x-kv.b500_U.mrobbie_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.mrobbie_sd1.4.bf16_r0', 'duo-x-kv.b100_U.beckham_sd1.4.bf16_r0', 'duo-x-kv.b250_U.beckham_sd1.4.bf16_r0', 'duo-x-kv.b500_U.beckham_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.beckham_sd1.4.bf16_r0', 'duo-x-kv.b100_U.rihanna_sd1.4.bf16_r0', 'duo-x-kv.b250_U.rihanna_sd1.4.bf16_r0', 'duo-x-kv.b500_U.rihanna_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.rihanna_sd1.4.bf16_r0', 'duo-x-kv.b100_U.obama_sd1.4.bf16_r0', 'duo-x-kv.b250_U.obama_sd1.4.bf16_r0', 'duo-x-kv.b500_U.obama_sd1.4.bf16_r0', 'duo-x-kv.b1000_U.obama_sd1.4.bf16_r0']

#         accelerate launch               \
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="naked_person"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.naked_sd1.4.bf16_r0"                          \
#             --target_prompt="naked person"                     \
#             --synonym_prompt=""                      \
#             --prior_prompt="dressed person"                              \
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
        
# ['duo-s_U.naked_sd1.4.bf16_r0']



#         accelerate launch               \
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="Margot_Robbie"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.mrobbie_sd1.4.bf16_r0"                          \
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
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="David_Beckham"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.beckham_sd1.4.bf16_r0"                          \
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
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="Rihanna"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.rihanna_sd1.4.bf16_r0"                          \
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
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="Barack_Obama"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.obama_sd1.4.bf16_r0"                          \
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
        
# ['duo-s_U.mrobbie_sd1.4.bf16_r0', 'duo-s_U.beckham_sd1.4.bf16_r0', 'duo-s_U.rihanna_sd1.4.bf16_r0', 'duo-s_U.obama_sd1.4.bf16_r0']


#         accelerate launch               \
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="a_painting_in_the_style_of_Picasso"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.picasso_sd1.4.bf16_r0"                          \
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
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="a_painting_in_the_style_of_Jackson_Pollock"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.pollock_sd1.4.bf16_r0"                          \
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
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="a_painting_in_the_style_of_Van_Gogh"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.vgogh_sd1.4.bf16_r0"                          \
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
#             --main_process_port 51234   \
#             unlearn-sd_custom.py               \
#             --project="SD-DPO_survival-no_prompt"               \
#             --mixed_precision="bf16"          \
#             --group=""                                          \
#             --config_dir="../data_root/generated/duo/config.json"    \
#             --config_name="a_painting_in_the_style_of_Claude_Monet"                      \
#             --data_dir="../data_root/generated/duo"                  \
#             --output_dir="../data_root/logs/duo/duo-s_U.cmonet_sd1.4.bf16_r0"                          \
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
        
# ['duo-s_U.picasso_sd1.4.bf16_r0', 'duo-s_U.pollock_sd1.4.bf16_r0', 'duo-s_U.vgogh_sd1.4.bf16_r0', 'duo-s_U.cmonet_sd1.4.bf16_r0']
#         accelerate launch               \
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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
#             --main_process_port 51234   \
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


