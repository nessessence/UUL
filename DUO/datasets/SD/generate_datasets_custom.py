# DUO
# Cpyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import os
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler, DDIMInverseScheduler
import numpy as np
from tqdm import tqdm
from PIL import Image
from nudenet import NudeDetector



def is_nsfw(img):
    nude_dict = nude_detector.detect(np.array(img))
    for item in nude_dict:
        if item["class"] in nudenet_labels:
            return True
    return False



parser = argparse.ArgumentParser(description="DUO dataset generator")
parser.add_argument("--save_dir", type=str, default="")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--gen_concept", type=str, default="ALL",
                    help="Which concept to generate: Nudity, Blood, Gun, Margot_Robbie, etc. Use ALL for everything.")


args = parser.parse_args()

device = args.device
weight_dtype = torch.float16
save_dir = args.save_dir

base_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=weight_dtype,
).to(device)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=weight_dtype,
).to(device)

base_pipe.safety_checker = None
pipe.safety_checker = None

nude_detector = NudeDetector()



exp_dict = {
    "Nudity": {
        "prompt": ["a naked white woman, realistic photo" for _ in range(64 // 2)]
        + ["a naked black woman, realistic photo" for _ in range(64 // 2)],
        "base_prompt": [
            "a dressed white woman, realistic photo" for _ in range(64 // 2)
        ]
        + ["a dressed black woman, realistic photo" for _ in range(64 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Blood": {
        "prompt": ["a white man, realistic photo, bleeding" for _ in range(32 // 2)]
        + ["a black man, realistic photo, bleeding" for _ in range(32 // 2)],
        "base_prompt": [
            "a white man, realistic photo, in comfort" for _ in range(32 // 2)
        ]
        + ["a black man, realistic photo, in comfort" for _ in range(32 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Suffer": {
        "prompt": ["a white man, realistic photo, suffering" for _ in range(32 // 2)]
        + ["a black man, realistic photo, suffering" for _ in range(32 // 2)],
        "base_prompt": [
            "a white man, realistic photo, enjoying" for _ in range(32 // 2)
        ]
        + ["a black man, realistic photo, enjoying" for _ in range(32 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Gun": {
        "prompt": ["a white man, realistic photo, with a gun" for _ in range(32 // 2)]
        + ["a black man, realistic photo, with a gun" for _ in range(32 // 2)],
        "base_prompt": [
            "a white man, realistic photo, with a peace" for _ in range(32 // 2)
        ]
        + ["a black man, realistic photo, with a peace" for _ in range(32 // 2)],
        "images": "unsafe",
        "base_images": "safe",
    },
    "Horror": {
        "prompt": "horror",
        "base_prompt": "comfort",
        "images": "unsafe",
        "base_images": "safe",
    },
}

### CUSTOM CONCEPTS ###
custom_concepts = [
    "Margot Robbie",
    "mickey mouse",
    "pad thai",
    "Barack Obama",
    "Donald Trump",
    "persian cat",
    "grumpy cat"
]

erase2general_concept = {"Margot Robbie": "person",
                            "mickey mouse": "cartoon character",
                            "pad thai": "food dish",
                            "Barack Obama": "person",
                            "Donald Trump": "person",
                            "persian cat": "cat",
                            "grumpy cat": "cat"
                            }

                            
for name in custom_concepts:
    general = erase2general_concept[name]
    key = name.replace(" ", "_")

    exp_dict[key] = {
        "prompt": [f"a photo of {name}" for _ in range(64)],
        "base_prompt": [f"a photo of a {general}" for _ in range(64)],
        "images": "unsafe",
        "base_images": "safe",
    }
#####
    
import json

with open("config.json", "w") as fp:
    json.dump(exp_dict, fp, indent=2)


nudenet_labels = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]
VIOLENCE_CONCEPTS = {"Blood", "Gun", "Suffer", "Horror"}



###############################################
# 9. GENERATE DATASETS
###############################################

for concept, prompt_dict in exp_dict.items():

    # filter by --gen_concept
    if args.gen_concept != "ALL":
        if concept.lower() != args.gen_concept.lower():
            continue
            
    print(f"\n=== Generating Concept: {concept} ===")

    prompt = prompt_dict["prompt"]
    base_prompt = prompt_dict["base_prompt"]
    num_images = len(prompt)

    unsafe_dir = os.path.join(save_dir, f"{concept}/unsafe")
    safe_dir = os.path.join(save_dir, f"{concept}/safe")
    os.makedirs(unsafe_dir, exist_ok=True)
    os.makedirs(safe_dir, exist_ok=True)

    ###############################################
    # 9A. GENERATE UNSAFE IMAGES
    ###############################################

    base_pipe.set_progress_bar_config(disable=True)

    for i in tqdm(range(num_images), desc=f"unsafe {concept}"):
        out_path = os.path.join(unsafe_dir, f"{i:02d}.jpg")
        if os.path.exists(out_path):
            continue

        # Nudity requires NSFW confirmation
        if concept == "Nudity":
            detected = False
            while not detected:
                img = base_pipe(prompt[i], num_images_per_prompt=1)[0][0]
                detected = is_nsfw(img)
        else:
            img = base_pipe(prompt[i], num_images_per_prompt=1)[0][0]

        img.save(out_path)

    ###############################################
    # 9B. GENERATE SAFE EDITED IMAGES
    ###############################################

    pipe.set_progress_bar_config(disable=True)

    for i in tqdm(range(num_images), desc=f"safe {concept}"):
        in_path = os.path.join(unsafe_dir, f"{i:02d}.jpg")
        out_path = os.path.join(safe_dir, f"{i:02d}.jpg")

        if os.path.exists(out_path):
            continue

        src = Image.open(in_path)

        # set editing strength
        # “strength” is one of the most important parameters in img2img and it directly controls how much the generated image is changed from the original.
        if concept in VIOLENCE_CONCEPTS:
            edit_strength = 0.85
        else:
            edit_strength = 0.75

        # Nudity requires removing NSFW via loop
        if concept == "Nudity":
            detected = True
            while detected:
                edit_img = pipe(
                    prompt=base_prompt[i],
                    negative_prompt=prompt[i],
                    image=src,
                    strength=edit_strength,
                    guidance_scale=7.5,
                )[0][0]
                detected = is_nsfw(edit_img)

            edit_img.save(out_path)
            continue

        # non-nudity: no NSFW loop
        edit_img = pipe(
            prompt=base_prompt[i],
            negative_prompt=prompt[i],
            image=src,
            strength=edit_strength,
            guidance_scale=7.5,
        )[0][0]
        edit_img.save(out_path)

print("\nAll done.")