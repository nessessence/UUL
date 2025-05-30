import os
import shutil

def insert_cfg_folder(base_dir="data_root/generated/model", cfg_scale="7.50"):
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        for checkpoint in os.listdir(model_path):
            checkpoint_path = os.path.join(model_path, checkpoint)
            if not os.path.isdir(checkpoint_path):
                continue

            for prompt in os.listdir(checkpoint_path):
                prompt_path = os.path.join(checkpoint_path, prompt)
                if not os.path.isdir(prompt_path):
                    continue

                # Destination: insert the cfg scale folder
                cfg_path = os.path.join(prompt_path, cfg_scale)
                os.makedirs(cfg_path, exist_ok=True)

                # Move all .png files into the new cfg_path
                for fname in os.listdir(prompt_path):
                    src = os.path.join(prompt_path, fname)
                    if os.path.isfile(src) and fname.lower().endswith(".png"):
                        shutil.move(src, os.path.join(cfg_path, fname))

if __name__ == "__main__":
    insert_cfg_folder()
