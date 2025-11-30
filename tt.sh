script="""
accelerate launch               \
    --main_process_port $port   \
    unlearn-sd.py               \
    --project="SD-DPO_survival-no_prompt"               \
    --group=""                                          \
    --config_dir="$base_dir/datasets/SD/config.json"    \
    --config_name="${config_name}"                      \
    --data_dir="$base_dir/datasets/SD"                  \
    --output_dir="${lora_dir}"                          \
    --target_prompt=""                     \
    --synonym_prompt=""                      \
    --prior_prompt=""                              \
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
"""