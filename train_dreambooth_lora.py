#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import time
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
# https://github.com/huggingface/diffusers/blob/v0.22.0/src/diffusers/loaders.py
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import wandb
import os.path as osp
from collections import defaultdict
from diffusers.utils.torch_utils import randn_tensor

from safetensors.torch import load_file

from esd.utils.surgery_util import custom_call,parse_generation_phase_parameter

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


print('diffusers version:', diffusers.__version__)
print('transformers version:', transformers.__version__)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0")

logger = get_logger(__name__)


def compute_mean_l2_param(unlearned_weights, original_weights, device=None):
    sq_sum = 0.0
    count = 0

    for k in unlearned_weights:


        w1 = unlearned_weights[k]
        w2 = original_weights[k]

        diff = w1 - w2
        sq_sum += diff.pow(2).sum()
        count += diff.numel()

    mean_l2 = torch.sqrt(sq_sum / count)
    return mean_l2, count

    

def convert_lora_weight(lora_pretrained_weight):
    
    # sign for different version
    if 'lora_A' in list(lora_pretrained_weight.keys())[0]:
        lora_pretrained_weight_ = {}
        print('detected NEW LoRA weight format, converting ....')
        for k,v in lora_pretrained_weight.items():
            assert 'unet' in k
            new_param_name = k.replace('unet','unet.unet')
            new_param_name = new_param_name.replace('lora_A','lora.down')
            new_param_name = new_param_name.replace('lora_B','lora.up')
            lora_pretrained_weight_[new_param_name] = v.clone()
        
            
    else:
        lora_pretrained_weight_ = lora_pretrained_weight
    return lora_pretrained_weight_

            
def count_images_in_dir(directory, extensions=None):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    return sum(
        1 for entry in os.scandir(directory)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in extensions
    )
    
def save_lora(
    unet=None,                 # accelerator.unwrap_model(unet) or None
    text_encoder=None,         # accelerator.unwrap_model(text_encoder) or None
    output_dir: str | None = None,
):
    """
    Save only the LoRA adapter weights.

    Parameters
    ----------
    unet : diffusers.UNet2DConditionModel | None
        UNet containing LoRA layers.
    text_encoder : transformers.PreTrainedModel | None
        Text-encoder containing LoRA layers.
    output_dir : str
        Directory where the *.bin files will be written.
    """
    if output_dir is None:
        raise ValueError("`output_dir` must be specified.")

    # Ensure directory exists (uses os / os.path only)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    unet_lora_layers  = None if unet is None else unet_lora_state_dict(unet)
    te_lora_layers    = None if text_encoder is None else text_encoder_lora_state_dict(text_encoder)
    # te_lora_layers = None
    if unet_lora_layers is None and te_lora_layers is None:
        raise ValueError("At least one of `unet` or `text_encoder` must be supplied.")

    LoraLoaderMixin.save_lora_weights(
        output_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=te_lora_layers,
    )

    
def resize_by_scale(image,scale=0.5):
    resized_image = image.resize( [int(scale * s) for s in image.size],  Image.Resampling.LANCZOS)
    return resized_image

def concatenate_images(image_list):
    num_rows = len(image_list)
    num_cols = len(image_list[0])  # Assuming all folders have the same number of images
    
    print(len(image_list))
    print(len(image_list[0]))
    
    img_width, img_height = image_list[0][0].size
    
    # Create a new image with the calculated size
    result_width = num_cols * img_width
    result_height = num_rows * img_height
    result_image = Image.new('RGB', (result_width, result_height))
    
    # Paste images into the result image
    for i, row_images in enumerate(image_list):
        for j, img in enumerate(row_images):
            x = j * img_width
            y = i * img_height
            result_image.paste(img, (x, y))
    
    return result_image


def save_token_embedding(text_encoder, placeholder_token, placeholder_token_id, accelerator, weight_path):
    logger.info(f"Saving Token Embeddings to {weight_path}")
    
    # Get the input embedding weights
    token_embeddings = text_encoder.get_input_embeddings().weight
    # Build the dictionary of token -> embedding
    learned_embeds_dict = {
        token: token_embeddings[token_id].detach().cpu()
        for token, token_id in zip(placeholder_token, placeholder_token_id)
    }
    torch.save(learned_embeds_dict, weight_path)
    
    
def load_token_embedding(text_encoder, tokenizer, weight_path):
    logger.info(f"Loading Token Embeddings from {weight_path}")
    # Load the saved token embeddings
    loaded_embeds_dict = torch.load(weight_path, weights_only=False)
    
    print(f"loaded_embeds_dict.keys(): {loaded_embeds_dict.keys()}")
    
    # Get the input embedding layer
    token_embeddings = text_encoder.get_input_embeddings()
    # Process each token
    for token, embed in loaded_embeds_dict.items():
        # Check if token already exists in tokenizer
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f'adding {token} to the tokenizer vocabs')
            # Token doesn't exist, add to tokenizer
            tokenizer.add_tokens([token])
            token_id = tokenizer.convert_tokens_to_ids(token)
            # Resize the embedding layer to match new vocab size
            text_encoder.resize_token_embeddings(len(tokenizer))
        # Set the embedding weight
        with torch.no_grad():
            print(f'loading embedding for {token}')
            token_embeddings.weight[token_id] = embed.to(token_embeddings.weight.device)
            
# def load_token_embedding(text_encoder,tokenizer, load_path):
    # if a placeholder does not exist, then add to the tokenizer first and then load
    # otherwise just replace the weight
    
    # write here #
    
    
    
    
    
    
    
    # learned_embeds_dict = {}
    # for i, ph_id in enumerate(placeholder_token_id):
    #     learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[ph_id]
    #     learned_embeds_dict[placeholder_token[i]] = learned_embeds.detach().cpu()
    # torch.save(learned_embeds_dict, save_path)
    
@torch.no_grad()
def log_validation(unet, text_encoder,tokenizer, args, accelerator, weight_dtype, epoch, log_label=None,save_image_path=None,gen_dtype=None, original_pretrained_weights=None, unlearned_weights=None):
    
    
    
        
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt: {args.validation_prompt}"
    )

    if gen_dtype is None: gen_dtype = weight_dtype
    print(f'gen_dtype: {gen_dtype}')
    # print(unet)
    # print(10*'#')
    # Initialize inference pipeline
    # pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,tokenizer=tokenizer, revision=args.revision, torch_dtype=args.gen_dtype)
    
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet).to(dtype=gen_dtype),
        text_encoder=accelerator.unwrap_model(text_encoder).to(dtype=gen_dtype),
        tokenizer=tokenizer,
        revision=args.revision,
        torch_dtype=gen_dtype,
    )
    
    # print(unet)
    # print(10*'#')
    # print(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.weight.dtype) #bfloat16
    # print(10*'#')
    # pipeline = DiffusionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     unet=unet,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     revision=args.revision,
    #     torch_dtype=gen_dtype,
    # )

    
    # pipeline.unet.eval()
    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        pipeline.scheduler.config.variance_type = variance_type

    if args.sampler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        print('apply DPM solver')
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline = pipeline.to(accelerator.device)
    # pipeline = pipeline.to(accelerator.device, dtype=gen_dtype)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
    apply_coco = False; use_coco30k = False; use_cocoval = False
    if '*coco30k' in args.validation_prompt[0]:
        apply_coco = True
        use_coco30k = True
        
        # expected: *coco30k.{n_sample}
        n_sample = int(args.validation_prompt[0].split('.')[-1])
        paired_prompt = torch.load("data_root/data/real_data/coco/id_caption_coco30k_seed123.pth")
        args.validation_prompt = [caption_ for id_,caption_ in paired_prompt[:n_sample]]
        args.num_validation_images = 1
    
    if '*cocoval' in args.validation_prompt[0]:
        apply_coco = True
        use_cocoval = True
        # expected: *cocoval.{n_sample}
        n_sample = int(args.validation_prompt[0].split('.')[-1])
        paired_prompt = torch.load("data_root/data/real_data/coco/id_caption_cocoval_seed123.pth")
        args.validation_prompt = [caption_ for id_,caption_ in paired_prompt[:n_sample]]
        args.num_validation_images = 1    

        # args.reinit_validation_generator = False ... use externally
        
    print('apply_coco:',apply_coco)

    cfg_scales = [ float(c) for c in args.cfg_scale.split(',')]
    # cfg_scales = [args.cfg_scale] if isinstance(args.cfg_scale, float) else args.cfg_scale

    images = []; index_images = []; prompt2images = defaultdict(list)
    
    
    use_new_route = True
    if apply_coco and use_new_route:
        cfg = cfg_scales[0]
        if use_coco30k:
            save_image_path_dir = osp.join(save_image_path,"coco30k_jpg", f"{cfg:.2f}")
            print(f'COCO30k: {n_sample} at:', save_image_path_dir)
        elif use_cocoval:
            save_image_path_dir = osp.join(save_image_path,"cocoval_jpg", f"{cfg:.2f}")
            print(f'COCOval: {n_sample} at:', save_image_path_dir)
            
        os.makedirs(save_image_path_dir,exist_ok=True)
        
        # args.gen_batch = 5
        for i in tqdm(range(0,len(args.validation_prompt),args.gen_batch ), total=len(args.validation_prompt)//args.gen_batch, disable=not apply_coco):
            prompts = args.validation_prompt[i: i + args.gen_batch]
            print(prompts)

            images_batch = pipeline(
                prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=cfg,
                generator=generator,
            ).images

            for rel_idx, image in enumerate(images_batch):
                abs_idx = i + rel_idx

                if save_image_path is not None:
                    fname = f"{abs_idx:04}.jpg"
                    image.save(osp.join(save_image_path_dir, fname))
                else:
                    img_prompt_tag = f'{prompt}_{cfg:.2f}'
                    images.append((img_prompt_tag, image))
                    index_images.append(abs_idx)
                    prompt2images[img_prompt_tag].append(image)
                            
            
    # old with coco
    else:
        for j,prompt in tqdm(enumerate(args.validation_prompt), total=len(args.validation_prompt)):
            
            
            for cfg in cfg_scales:
                print(f'prompt: {prompt} cfg: {cfg:.2f} neg_prompt: {args.negative_prompt is not None }')
                
                # if apply_coco:
                #     print(f'{j+1} / {n_sample}')
                skip_already_generated = False
                if save_image_path is not None:
                    if apply_coco: 
                        save_image_path_dir = osp.join(save_image_path,"coco30k", f"{cfg:.2f}")
                        if skip_already_generated: 
                            if os.path.exists(save_image_path_dir) and count_images_in_dir(save_image_path_dir) >= len(args.validation_prompt):  # TODO: should count only images
                                logger.info(f"Skipping COCO as already exist in {save_image_path_dir} with {len(args.validation_prompt)} images")
                                print('ending generation')
                                exit()
                    else:
                        if args.use_custom_pipeline and prompt.startswith('*Ph.'):
                            save_image_path_dir = osp.join(save_image_path,f"{prompt.split('*Ph.')[-1]}", f"{cfg:.2f}")
                            
                        elif args.negative_prompt is not None:
                            save_image_path_dir = osp.join(save_image_path,f"{prompt}_neg", f"{cfg:.2f}")
                            
                        elif prompt == '':
                            save_image_path_dir = osp.join(save_image_path,f"uncond", f"{cfg:.2f}")
                            
                        # cce step tag
                        elif ('v0' in prompt or 'cce0' in prompt) and args.load_token_embedding_step is not None:
                            save_image_path_dir = osp.join(save_image_path,f"{prompt}-{args.load_token_embedding_step}", f"{cfg:.2f}")
                            
                        else:
                            save_image_path_dir = osp.join(save_image_path,prompt, f"{cfg:.2f}")
                        if skip_already_generated: 
                            if os.path.exists(save_image_path_dir) and count_images_in_dir(save_image_path_dir) >= args.num_validation_images:  # TODO: should count only images
                                logger.info(f"Skipping {prompt} as  already exist in {save_image_path_dir} with {len(args.num_validation_images)} images")
                                continue
                    os.makedirs(save_image_path_dir,exist_ok=True)
                    print("save_image_path_dir:",save_image_path_dir)
                        
                        
                if args.reinit_validation_generator:
                    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
                
                # hacky:  always generate the first image (usually the target) 100 samples
                if j == 0:
                    num_images = 100
                else: num_images = args.num_validation_images
                
                
                batch_size = args.gen_batch #args.gen_batch

                for i in tqdm(range(0, num_images, batch_size),disable= apply_coco):
                    batch_indices = range(i, min(i + batch_size, num_images))
                    
                    prompts = [prompt] * len(batch_indices)
                    # generators = [torch.Generator(device=accelerator.device).manual_seed(args.seed + idx) for idx in batch_indices]
                    if args.use_custom_pipeline and prompt.startswith('*Ph.'):
                        # custom generation phase parameters
                        assert original_pretrained_weights is not None and unlearned_weights is not None
                        generation_phase_parameter,simplified_phase_parameter = parse_generation_phase_parameter(prompt,  orginal_pretrained_weight=original_pretrained_weights,unlearned_weight=unlearned_weights)
                        print(f'custom generation phase parameters: {simplified_phase_parameter}')
                        images_batch = pipeline(
                            [""]* len(batch_indices), # dummy
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=cfg,
                            generator=generator,
                            generation_phase_parameter=generation_phase_parameter,
                        ).images
                        
                    
                    

                    elif args.negative_prompt is not None:
                        negative_prompts = [args.negative_prompt] * len(batch_indices)
                        images_batch = pipeline(
                            prompts,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=cfg,
                            negative_prompt=negative_prompts,
                            generator=generator,
                        ).images
                    else:
                        images_batch = pipeline(
                            prompts,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=cfg,
                            generator=generator,
                        ).images

                    for rel_idx, image in enumerate(images_batch):
                        abs_idx = i + rel_idx

                        if save_image_path is not None:
                            fname = f"{j:04}.png" if apply_coco else f"{abs_idx:04}.png"
                            image.save(osp.join(save_image_path_dir, fname))
                        else:
                            img_prompt_tag = f'{prompt}_{cfg:.2f}'
                            images.append((img_prompt_tag, image))
                            index_images.append(abs_idx)
                            prompt2images[img_prompt_tag].append(image)
                    
            # for i in tqdm(range(args.num_validation_images)):
                
            #     # dummy_latents = randn_tensor( (1, 4, 64, 64),device=accelerator.device, generator=generator)
            #     image = pipeline(prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.cfg_scale, generator=generator).images[0]
            #     if save_image_path is not None:
            #         if apply_coco:
            #             img_path = osp.join(save_image_path_dir,f"{j:04}.png")
            #         else:
            #             img_path = osp.join(save_image_path_dir,f"{i:04}.png")
            #         image.save(img_path)
                
            #         # img_path = osp.join(save_image_path_dir, f"{i:04}.jpg")  # i is your sample index
            #         # image.save(img_path,format="JPEG")
            #     else:
                
            #         images.append((prompt, image))
            #         index_images += [i]
                    
            #         prompt2images[prompt] += [image]
            
            
    if save_image_path is not None: return
            
        # for _ in range(args.num_validation_images):
        #     with torch.cuda.amp.autocast():
        #         image = pipeline(**pipeline_args, generator=generator).images[0]
        #         images.append(image)

    log_label = '' if log_label is None else f"{log_label}_"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            # tracker.log(
            #         {
            #             f"{log_label}validation": [
            #                 wandb.Image(resize_by_scale(image,scale=0.50), caption=f"{prompt} : {i}", file_type='jpg') for (i, (prompt, image)) in zip(index_images,images)
            #             ]
            #         }
            #     )
            ## concat image
            # concat_list = [prompt2images[prompt] for prompt in args.validation_prompt]
            prompt_tags = list(prompt2images.keys())
            concat_list = [prompt2images[prompt_tag] for prompt_tag in prompt_tags]
            concated_image = concatenate_images(concat_list )
            tracker.log(
                {
                    f"{log_label}concat": [
                        wandb.Image(resize_by_scale(concated_image,scale=0.50), caption=prompt, file_type='jpg')
                    ]
                })
    
                
            # tracker.log({
            #     "validation": [
            #         wandb.Image(resize_by_scale(image,scale=0.50), caption=f"{i}: {args.validation_prompt}")
            #         for i, image in enumerate(images)
            #     ]
            # })
            
    if gen_dtype != weight_dtype:
        unet = unet.to(dtype=weight_dtype)
        text_encoder = text_encoder.to(dtype=weight_dtype)

    del pipeline
    torch.cuda.empty_cache()
    
    

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- {'stable-diffusion' if isinstance(pipeline, StableDiffusionPipeline) else 'if'}
- {'stable-diffusion-diffusers' if isinstance(pipeline, StableDiffusionPipeline) else 'if-diffusers'}
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    
    
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
        
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=10,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help=("The dimension of the LoRA update matrices."),
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--target_lora_modules",
        type=str,
        nargs="+",
        default=["to_k", "to_v"],
        help="Subset of attention modules to apply LoRA to: choose from [to_q, to_k, to_v, to_out, add_k_proj, add_v_proj]"
    )
        
    parser.add_argument(
        "--target_lora_layers",
        type=str,
        nargs="+",
        default=["cross"],
        choices=["cross", "self"],
        help='Which attention types to apply LoRA to. Choose from ["cross", "self"].',
    )
    parser.add_argument("--run_note",type=str,default=None)  




    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )


    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument("--gen_batch",type=int,default=25,)


    parser.add_argument("--flip_p",type=float,default=0.5,)
    parser.add_argument("--num_inference_steps",type=int,default=50,)
    parser.add_argument("--cfg_scale",default=3.0)

    parser.add_argument( "--test_run",action="store_true")
    
    
    # for image generation only
    parser.add_argument( "--gen_image_path",type=str,default=None)
    parser.add_argument( "--load_lora_weight_path",type=str,default=None)
    parser.add_argument( "--load_unet_weight_path",type=str,default=None) # many unlearned model, UCE, ESD, 
    
    parser.add_argument( "--load_token_embedding_path",type=str,default=None)
    parser.add_argument( "--load_token_embedding_step",type=int,default=None)
    
    
    parser.add_argument( "--gen_dtype",type=str,default="fp16")
    
    
    parser.add_argument( "--load_pretrained_lora_weight_path",type=str,default=None)
    parser.add_argument( "--load_pretrained_token_embedding_path",type=str,default=None)

    
    
    parser.add_argument("--wait_weight",action="store_true",default=False,)
    
    parser.add_argument("--learning_rate_ti",type=float,default=None,)
    parser.add_argument("--learning_rate_lora",type=float,default=None,)
    parser.add_argument("--learning_rate_lora_text_encoder",type=float,default=None,)
    parser.add_argument("--sampler",type=str,default="DDIM",)

    # Hacked debt
    parser.add_argument("--donot_reinit_validation_generator",action="store_true")

    parser.add_argument("--use_custom_pipeline",action="store_true")

    
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, help="A token to use as initializer word."
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    Now also supports *lists* of instance folders/prompts that are paired by index.
    """

    def __init__(
        self,
        instance_data_root,          # str | Path | List[str|Path]
        instance_prompt,             # str | List[str]
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        flip_p=0.0,
    ):
        # ───────────────────────────── basic attrs ─────────────────────────────
        self.size        = size
        self.center_crop = center_crop
        self.tokenizer   = tokenizer
        self.encoder_hidden_states              = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.flip_p               = flip_p

        # ───────────────────────── special-case: multi data/prompts ────────────
        is_multi = (
            isinstance(instance_data_root, (list, tuple))
            and isinstance(instance_prompt, (list, tuple))
        )

        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

        if is_multi:
            if len(instance_data_root) != len(instance_prompt):
                raise ValueError(
                    "`instance_data_root` and `instance_prompt` must have the same length "
                    f"(got {len(instance_data_root)} vs {len(instance_prompt)})."
                )

            self.instance_images_path = []
            self.instance_prompts_per_image = []

            for root, prompt in zip(instance_data_root, instance_prompt):
                root = Path(root)
                if not root.exists():
                    raise ValueError(f"Instance images root '{root}' doesn't exist.")
                images_here = [
                    p for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
                self.instance_images_path += images_here                    # ← using “+= []”
                
                print(f"Found {len(images_here)} images in {root} with prompt: {prompt}")
                self.instance_prompts_per_image += [prompt] * len(images_here)

            self.num_instance_images = len(self.instance_images_path)
            self._length             = self.num_instance_images
            self.instance_prompt     = None      # signals per-image prompts downstream
        else:
            # ───────────── original single-folder path (unchanged) ────────────
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exist.")

            self.instance_images_path = [
                p
                for p in self.instance_data_root.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            self.num_instance_images = len(self.instance_images_path)
            self.instance_prompt     = instance_prompt
            self._length             = self.num_instance_images

        # ───────────────────────── class images (unchanged) ───────────────────
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = (
                min(len(self.class_images_path), class_num)
                if class_num is not None
                else len(self.class_images_path)
            )
            self._length = max(self._length, self.num_class_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        # ───────────────────────── transforms (unchanged) ─────────────────────
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                *([transforms.RandomHorizontalFlip(p=flip_p)] if flip_p > 0 else []),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # ─────────────────────────── torch-Dataset API ────────────────────────────
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        real_idx = index % self.num_instance_images

        # ---------------- instance image ----------------
        instance_image = Image.open(self.instance_images_path[real_idx])
        instance_image = exif_transpose(instance_image)
        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        # ---------------- instance prompt ----------------
        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            if self.instance_prompt is None:                     # multi-prompt mode
                prompt_str = self.instance_prompts_per_image[real_idx]
            else:                                                # single-prompt mode
                prompt_str = self.instance_prompt
                
            # print(real_idx,prompt_str)

            text_inputs = tokenize_prompt(
                self.tokenizer, prompt_str, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"]     = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        # ---------------- class (prior-preservation) part (unchanged) ----------
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"]     = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example

# class DreamBoothDataset(Dataset):
#     """
#     A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
#     It pre-processes the images and the tokenizes prompts.
#     """

#     def __init__(
#         self,
#         instance_data_root,
#         instance_prompt,
#         tokenizer,
#         class_data_root=None,
#         class_prompt=None,
#         class_num=None,
#         size=512,
#         center_crop=False,
#         encoder_hidden_states=None,
#         class_prompt_encoder_hidden_states=None,
#         tokenizer_max_length=None,
#         flip_p=0.0
#     ):
#         self.size = size
#         self.center_crop = center_crop
#         self.tokenizer = tokenizer
#         self.encoder_hidden_states = encoder_hidden_states
#         self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
#         self.tokenizer_max_length = tokenizer_max_length
#         self.flip_p = flip_p

#         self.instance_data_root = Path(instance_data_root)
#         if not self.instance_data_root.exists():
#             raise ValueError("Instance images root doesn't exists.")

#         # self.instance_images_path = list(Path(instance_data_root).iterdir())
#         # self.instance_images_path = [p for p in Path(instance_data_root).iterdir() if p.is_file()]
        
#         IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
#         self.instance_images_path = [p for p in Path(instance_data_root).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

#         self.num_instance_images = len(self.instance_images_path)
#         self.instance_prompt = instance_prompt
#         self._length = self.num_instance_images

#         if class_data_root is not None:
#             self.class_data_root = Path(class_data_root)
#             self.class_data_root.mkdir(parents=True, exist_ok=True)
#             self.class_images_path = list(self.class_data_root.iterdir())
#             if class_num is not None:
#                 self.num_class_images = min(len(self.class_images_path), class_num)
#             else:
#                 self.num_class_images = len(self.class_images_path)
#             self._length = max(self.num_class_images, self.num_instance_images)
#             self.class_prompt = class_prompt
#         else:
#             self.class_data_root = None

#         self.image_transforms = transforms.Compose(
#             [
#                 transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
#                 transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
#                 *([transforms.RandomHorizontalFlip(p=flip_p)] if flip_p > 0 else []),

#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )
        
#         print(f'image transformers:\n{self.image_transforms}')

#     def __len__(self):
#         return self._length

#     def __getitem__(self, index):
#         example = {}
#         instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
#         instance_image = exif_transpose(instance_image)

#         if not instance_image.mode == "RGB":
#             instance_image = instance_image.convert("RGB")
#         example["instance_images"] = self.image_transforms(instance_image)

#         if self.encoder_hidden_states is not None:
#             example["instance_prompt_ids"] = self.encoder_hidden_states
#         else:
#             text_inputs = tokenize_prompt(
#                 self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
#             )
#             example["instance_prompt_ids"] = text_inputs.input_ids
#             example["instance_attention_mask"] = text_inputs.attention_mask

#         if self.class_data_root:
#             class_image = Image.open(self.class_images_path[index % self.num_class_images])
#             class_image = exif_transpose(class_image)

#             if not class_image.mode == "RGB":
#                 class_image = class_image.convert("RGB")
#             example["class_images"] = self.image_transforms(class_image)

#             if self.class_prompt_encoder_hidden_states is not None:
#                 example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
#             else:
#                 class_text_inputs = tokenize_prompt(
#                     self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
#                 )
#                 example["class_prompt_ids"] = class_text_inputs.input_ids
#                 example["class_attention_mask"] = class_text_inputs.attention_mask

#         return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    
    if args.use_custom_pipeline:
        print(f'using Custom Stable Diffusion Call from: custom_call')
        StableDiffusionPipeline.__call__ = custom_call
    
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f'weight dtype: {weight_dtype}')
    
    
    fast_track = True
    if args.gen_image_path is not None and fast_track:
        print('entering fastrack generation')
        torch.cuda.empty_cache()

        print(f'entering image generation, save image to: {args.gen_image_path}')
        # pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype)
        pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=args.gen_dtype)

        if args.load_pretrained_lora_weight_path is not None and args.load_pretrained_lora_weight_path:
            print('loading LoRA into UNet ....')
            print(f'LoRA path: {args.load_pretrained_lora_weight_path}')
        
            dummy_pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, unet=pipeline.unet, text_encoder=pipeline.text_encoder, vae=pipeline.vae, revision=args.revision)
            
            lora_pretrained_weight_ = load_file(osp.join(args.load_pretrained_lora_weight_path, "pytorch_lora_weights.safetensors"))
            lora_pretrained_weight = convert_lora_weight(lora_pretrained_weight_)
            pipeline.load_lora_weights(lora_pretrained_weight)
            
            
            # dummy_pipeline.load_lora_weights(args.load_pretrained_lora_weight_path, weight_name="pytorch_lora_weights.safetensors")

            dummy_pipeline.fuse_lora()
            print('Fused LoRA  ....')
                
        
        if args.load_unet_weight_path is not None:
            print('loading UNet weight from: ', args.load_unet_weight_path)
            if '.safetensor' in args.load_unet_weight_path:
                
                compute_weight_diff = False
                
                if compute_weight_diff:
                    unlearned_weights = load_file(args.load_unet_weight_path)
                    # only weight that are the same
                    original_weights = pipeline.unet.state_dict() #
                    weight_diff,_ = compute_mean_l2_param(unlearned_weights, original_weights)
                    
                    print('load_unet_weight_path:', args.load_unet_weight_path)
                    print(f'mean L2 weight diff: {weight_diff.item()}')
                    
                    pipeline.unet.load_state_dict(unlearned_weights, strict=False)
                    
                
                else:
                    pipeline.unet.load_state_dict(load_file(args.load_unet_weight_path), strict=False)
            else:
                pipeline.unet.load_state_dict(torch.load(args.load_unet_weight_path), strict=False)
            print('UNet weight loaded (for generation)')
            
            
            pipeline.unet.eval()
        else:
            print('not loading UNet weight')            
            
        
        if args.load_lora_weight_path is not None and args.lora_rank is not None and args.lora_rank > 0:
            # load attention processors
            print(25*"#")
            print('loading LoRA weight')
            
            lora_pretrained_weight_ = load_file(osp.join(args.load_lora_weight_path, "pytorch_lora_weights.safetensors"))
            lora_pretrained_weight = convert_lora_weight(lora_pretrained_weight_)
            pipeline.load_lora_weights(lora_pretrained_weight)
        
        
            # pipeline.load_lora_weights(args.load_lora_weight_path, weight_name="pytorch_lora_weights.safetensors")
            
            print(f'generating images from: lora{args.load_lora_weight_path}')
            
        else: print('not loading loRA weight')
        
        
        # print( args.load_token_embedding_path,osp.isfile(args.load_token_embedding_path))
        if args.load_token_embedding_path is not None and args.placeholder_token is not None and  osp.isdir(args.load_token_embedding_path):
            print('loading token embedding')
            file_name = f'token_embedding-{args.load_token_embedding_step}.pt' if args.load_token_embedding_step is not None else 'token_embedding.pt'
            # print(f'loading token embedding from: {osp.join(args.load_token_embedding_path,file_name)}')
            load_token_embedding(pipeline.text_encoder, pipeline.tokenizer, osp.join(args.load_token_embedding_path,file_name))
            
        
        elif args.load_token_embedding_path is not None:
            print('loading token embedding')
            print(f'loading token embedding from: {args.load_token_embedding_path}')
            load_token_embedding(pipeline.text_encoder, pipeline.tokenizer, args.load_token_embedding_path)
            
        
        log_validation(
            unet=pipeline.unet,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            gen_dtype=args.gen_dtype,
            epoch=0,
            log_label='image generation',
            save_image_path = args.gen_image_path)
        
        exit()
                    
                    
    
    
    
    


    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_lora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            # print(f'unet_lora_layers_to_save: {unet_lora_layers_to_save}')
            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
        )
        
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        
        project = "uul"
        if args.test_run or args.gen_image_path is not None:
            project = 'test'
        display_name = osp.basename(args.output_dir)
        wandb.init(project=project, name=display_name)

        
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None
        
    if vae is not None:
        print(f"VAE loaded from {args.pretrained_model_name_or_path}/vae")
    else:
        print(f"VAE not found in {args.pretrained_model_name_or_path}/vae, skipping VAE loading.")
    # exit()
    
    # unet = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=args.gen_dtype).unet
    
    
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
    )
    
    
    if args.load_unet_weight_path is not None:
        print(f'loading unet weights from {args.load_unet_weight_path}')
        if '.safetensor' in args.load_unet_weight_path:
            unet.load_state_dict(load_file(args.load_unet_weight_path), strict=False)
        else:
            unet.load_state_dict(torch.load(args.load_unet_weight_path), strict=False)
        print('unet weights loaded')


    # load pretrained LoRA weights if provided
    if args.load_pretrained_lora_weight_path is not None and args.load_pretrained_lora_weight_path:
        print('loading LoRA into UNet ....')
        print(f'LoRA path: {args.load_pretrained_lora_weight_path}')
    
        dummy_pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, unet=unet, text_encoder=text_encoder, vae=vae, revision=args.revision)
        

        lora_pretrained_weight_ = load_file(osp.join(args.load_pretrained_lora_weight_path, "pytorch_lora_weights.safetensors"))
        lora_pretrained_weight = convert_lora_weight(lora_pretrained_weight_)
        # print('loaded LoRA weights')
        dummy_pipeline.load_lora_weights(lora_pretrained_weight)
        # dummy_pipeline.load_lora_weights(args.load_pretrained_lora_weight_path, weight_name="pytorch_lora_weights.safetensors")

        dummy_pipeline.fuse_lora()
        
        print('Fused LoRA  ....')
            


    
    # ───────────────────────── textual-inversion token setup ─────────────────────────
    if args.placeholder_token is not None:
        # Split on commas → strip whitespace → drop empties
        placeholder_tokens = [t.strip() for t in args.placeholder_token.split(",") if t.strip()]
        initializer_tokens = (
            [t.strip() for t in args.initializer_token.split(",") if t.strip()]
            if (args.initializer_token is not None and args.initializer_token != "")
            else []
        )

        print(
            f"Applying textual inversion\n"
            f"  placeholder tokens : {placeholder_tokens}\n"
            f"  initializer tokens : {initializer_tokens if initializer_tokens else '〈none〉'}"
        )

        # ---------------- add new placeholder tokens to the tokenizer ----------------
        tokens_to_add = [
            tok for tok in placeholder_tokens
            if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id
        ]
        
        if tokens_to_add:
            num_added_tokens = tokenizer.add_tokens(tokens_to_add)
            print(f"Added {num_added_tokens} new token(s): {tokens_to_add}")
            text_encoder.resize_token_embeddings(len(tokenizer))
        else:
            print("All placeholder tokens already present in the tokenizer.")

        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)


        # ---------------- initialise embeddings ----------------
        if not initializer_tokens:
            print("No initializer token(s) provided → skipping embedding initialisation.")
        else:
            # --- normalise initializer list to match #placeholders ---
            if len(initializer_tokens) == 1:
                initializer_tokens = initializer_tokens * len(placeholder_tokens)
            elif len(initializer_tokens) != len(placeholder_tokens):
                raise ValueError(
                    "Number of initializer tokens must be either 1 or equal to #placeholder tokens "
                    f"(got {len(initializer_tokens)} vs {len(placeholder_tokens)})."
                )
            # Convert each initializer token to a single-token id
            init_token_ids = []
            for tok in initializer_tokens:
                # Handle special "uncond" keyword for unconditional (empty) prompt
                if tok.lower() == "uncond":
                    # Use EOS token (what fills unconditional prompts in CFG)
                    eos_token_id = tokenizer.eos_token_id
                    if eos_token_id is None:
                        # Fallback: encode empty string WITH special tokens and use second token (EOS)
                        ids = tokenizer.encode("", add_special_tokens=True)
                        if len(ids) > 1:
                            eos_token_id = ids[1]  # Second token is EOS
                        elif len(ids) == 1:
                            eos_token_id = ids[0]  # Fallback to first token if only one
                        else:
                            print(f"Warning: Could not determine EOS token, skipping initialization")
                            init_token_ids.append(None)
                            continue
                    init_token_ids.append(eos_token_id)
                    print(f"  Using EOS token (ID: {eos_token_id}) for 'uncond' initialization")
                elif tok.lower() == "random":
                    # Mark for random initialization
                    init_token_ids.append("random")
                    print(f"  Will use random initialization for this token")
                else:
                    ids = tokenizer.encode(tok, add_special_tokens=False)
                    if len(ids) != 1:
                        raise ValueError(f"Initializer token '{tok}' is not a single tokenizer token.")
                    init_token_ids.append(ids[0])

            # Copy embeddings one-by-one
            token_embeds = text_encoder.get_input_embeddings().weight.data
            
            # Calculate actual embedding std for random initialization
            clip_embedding_std = token_embeds.std().item()
            print(f"Measured CLIP embedding std: {clip_embedding_std:.4f}")
            
            with torch.no_grad():
                for ph_id, init_id, ph_tok, init_tok in zip(
                    placeholder_token_ids, init_token_ids, placeholder_tokens, initializer_tokens
                ):
                    if init_id is None:
                        continue
                    elif init_id == "random":
                        # Random initialization with measured CLIP std
                        embedding_dim = token_embeds.shape[1]
                        generator = torch.Generator(device=token_embeds.device).manual_seed(args.seed)
                        token_embeds[ph_id] = torch.randn(embedding_dim, generator=generator, device=token_embeds.device) * clip_embedding_std
                        print(f"  ↳ randomly initialized '{ph_tok}' with std={clip_embedding_std:.4f} (seed={args.seed})")
                    else:
                        token_embeds[ph_id] = token_embeds[init_id].clone()
                        print(f"  ↳ initialised '{ph_tok}' from '{init_tok}' (token ID: {init_id})")

        # # ---------------- initialise embeddings ----------------
        # if not initializer_tokens:
        #     print("No initializer token(s) provided → skipping embedding initialisation.")
        # else:
        #     # --- normalise initializer list to match #placeholders ---
        #     if len(initializer_tokens) == 1:
        #         initializer_tokens = initializer_tokens * len(placeholder_tokens)
        #     elif len(initializer_tokens) != len(placeholder_tokens):
        #         raise ValueError(
        #             "Number of initializer tokens must be either 1 or equal to #placeholder tokens "
        #             f"(got {len(initializer_tokens)} vs {len(placeholder_tokens)})."
        #         )

        #     # Convert each initializer token to a single-token id
        #     init_token_ids = []
        #     for tok in initializer_tokens:
        #         # Handle special "uncond" keyword for unconditional (empty) prompt
        #         if tok.lower() == "uncond":
        #             # Encode empty string to get unconditional token
        #             ids = tokenizer.encode("", add_special_tokens=False)
        #             if len(ids) == 0:
        #                 # If empty string produces no tokens, use padding token or skip
        #                 print(f"Warning: 'uncond' produced no tokens, skipping initialization for this token")
        #                 init_token_ids.append(None)
        #                 continue
        #             init_token_ids.append(ids[0])
        #         else:
        #             ids = tokenizer.encode(tok, add_special_tokens=False)
        #             if len(ids) != 1:
        #                 raise ValueError(f"Initializer token '{tok}' is not a single tokenizer token.")
        #             init_token_ids.append(ids[0])

        #     # Copy embeddings one-by-one
        #     token_embeds = text_encoder.get_input_embeddings().weight.data
        #     with torch.no_grad():
        #         for ph_id, init_id, ph_tok, init_tok in zip(
        #             placeholder_token_ids, init_token_ids, placeholder_tokens, initializer_tokens
        #         ):
        #             if init_id is None:
        #                 continue
        #             token_embeds[ph_id] = token_embeds[init_id].clone()
        #             print(f"  ↳ initialised '{ph_tok}' from '{init_tok}'")

     


    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    unet.requires_grad_(False)


    if args.placeholder_token is not None:
        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)
        
    
    # # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # # as these weights are only used for inference, keeping weights in full precision is not required.
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    # print(f'weight dtype: {weight_dtype}')

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x up blocks) = 18
    # => 32 layers


    # Set correct LoRA layers
    print_lora_layers = False
    unet_lora_parameters = []

    # Helper to build kwargs for LoRALinearLayer
    def _lora_kwargs(in_f, out_f):
        return {
            "in_features": in_f,
            "out_features": out_f,
            "rank": args.lora_rank,
            **({"network_alpha": args.lora_alpha} if args.lora_alpha is not None else {}),
        }

    if args.lora_rank is not None and args.lora_rank > 0:
        for attn_processor_name, attn_processor in unet.attn_processors.items():

            if ("attn1" in attn_processor_name and "self" not in args.target_lora_layers) or \
            ("attn2" in attn_processor_name and "cross" not in args.target_lora_layers):
                if print_lora_layers:
                    print(f'skipping layer: {attn_processor_name}')
                continue

            # Parse the attention module.
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            if print_lora_layers:
                print(f'attn_processor_name: {attn_processor_name} - {attn_module}')

            if "to_q" in args.target_lora_modules:
                if print_lora_layers:
                    print('LoRA q')
                attn_module.to_q.set_lora_layer(
                    LoRALinearLayer(**_lora_kwargs(attn_module.to_q.in_features, attn_module.to_q.out_features))
                )
                unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())

            if "to_k" in args.target_lora_modules:
                if print_lora_layers:
                    print('LoRA k')
                attn_module.to_k.set_lora_layer(
                    LoRALinearLayer(**_lora_kwargs(attn_module.to_k.in_features, attn_module.to_k.out_features))
                )
                unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())

            if "to_v" in args.target_lora_modules:
                if print_lora_layers:
                    print('LoRA v')
                attn_module.to_v.set_lora_layer(
                    LoRALinearLayer(**_lora_kwargs(attn_module.to_v.in_features, attn_module.to_v.out_features))
                )
                unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())

            if "to_out" in args.target_lora_modules:
                if print_lora_layers:
                    print('LoRA out')
                attn_module.to_out[0].set_lora_layer(
                    LoRALinearLayer(**_lora_kwargs(attn_module.to_out[0].in_features, attn_module.to_out[0].out_features))
                )
                unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):

                if "add_k_proj" in args.target_lora_modules:
                    if print_lora_layers:
                        print('LoRA add k')
                    attn_module.add_k_proj.set_lora_layer(
                        LoRALinearLayer(**_lora_kwargs(attn_module.add_k_proj.in_features, attn_module.add_k_proj.out_features))
                    )
                    unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())

                if "add_v_proj" in args.target_lora_modules:
                    if print_lora_layers:
                        print('LoRA add v')
                    attn_module.add_v_proj.set_lora_layer(
                        LoRALinearLayer(**_lora_kwargs(attn_module.add_v_proj.in_features, attn_module.add_v_proj.out_features))
                    )
                    unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.lora_rank)
        # text_lora_parameters = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=weight_dtype, rank=args.lora_rank)

    # print(text_lora_parameters)
    # print(text_lora_parameters[-1])
    # print(type(text_lora_parameters))
    
    # print(text_encoder)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    # Optimizer creation
    
    params_to_optimize = []
    if args.learning_rate_lora is not None and args.learning_rate_ti is not None:
        print('using a separate lr for lora and ti')
        # LoRA parameters
        if args.lora_rank is not None and args.lora_rank > 0:
            
            if args.train_text_encoder and args.learning_rate_lora_text_encoder :
                
                print(f"unet_lr: {args.learning_rate_lora}, text_encoder_lr: {args.learning_rate_lora_text_encoder}")
                params_to_optimize.append({
                        "params": unet_lora_parameters,
                        "lr": args.learning_rate_lora
                    })
                
                params_to_optimize.append({
                        "params": text_lora_parameters,
                        "lr": args.learning_rate_lora_text_encoder
                    })
            else:
                print("using lora as learnable parameters")
                params_lora = list(
                    itertools.chain(unet_lora_parameters, text_lora_parameters)
                    if args.train_text_encoder else unet_lora_parameters
                )
                if args.learning_rate_lora is not None:
                    params_to_optimize.append({
                        "params": params_lora,
                        "lr": args.learning_rate_lora
                    })
                else:
                    params_to_optimize.append({"params": params_lora})

        # Token embeddings (TI)
        if args.placeholder_token is not None:
            print("adding token embeddings as learnable parameters")
            params_ti = list(text_encoder.get_input_embeddings().parameters())
            if args.learning_rate_ti is not None:
                params_to_optimize.append({
                    "params": params_ti,
                    "lr": args.learning_rate_ti
                })
            else:
                params_to_optimize.append({"params": params_ti})
                
    else:
        # LoRA
        params_lora = []
        if args.lora_rank is not None and  args.lora_rank > 0:
            print('adding lora as learnable parameters')
            params_lora = (
                itertools.chain(unet_lora_parameters, text_lora_parameters)
                if args.train_text_encoder
                else unet_lora_parameters
            )
            print(f"unet_lora_parameters dtype: {next(iter(unet_lora_parameters)).dtype}")
        # TI
        params_ti = []
        if args.placeholder_token is not None:
            print('adding token embeddings as learnable parameters')
            params_ti = list(text_encoder.get_input_embeddings().parameters())
            
            
        params_to_optimize = params_lora + params_ti
    
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # obsolete
    pre_computed_encoder_hidden_states = None
    validation_prompt_encoder_hidden_states = None
    validation_prompt_negative_prompt_embeds = None
    pre_computed_class_prompt_encoder_hidden_states = None



    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
        flip_p=args.flip_p
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth-lora", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    
    
    
    save_step0 = True
    if save_step0 and not args.gen_image_path:
        print('save epoch 0 applied')
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        # accelerator.save_state(save_path) # lora also saved here
        save_lora(
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder) if args.train_text_encoder else None,
            output_dir=os.path.join(save_path)
        )
        logger.info(f"Saved state to {save_path}")
        
        # save ti
        # placeholder_tokens = [args.placeholder_token]
        # placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens) # also
        # save_token_embedding(accelerator.unwrap_model(text_encoder), placeholder_tokens, placeholder_token_ids, accelerator, osp.join(save_path,'token_embedding.pt'))
                       
                       
        if args.placeholder_token is not None:
            # ─────────────────────── save textual-inversion embeddings ───────────────────────
            # Accept either a single token or a comma-separated list
            placeholder_tokens = [t.strip() for t in args.placeholder_token.split(",") if t.strip()]

            # Convert each placeholder token to its ID (list-compatible already)
            placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

            save_token_embedding(
                accelerator.unwrap_model(text_encoder),
                placeholder_tokens,
                placeholder_token_ids,
                accelerator,
                osp.join(save_path, "token_embedding.pt"),
            )
            print(f"Saved embeddings for {len(placeholder_tokens)} token(s): {placeholder_tokens}")

        
    if args.gen_image_path is not None: 
        print('no training ... setting epoch to 0')
        args.num_train_epochs = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            
            # print(text_lora_parameters[-1].mean())
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                if args.pre_compute_text_embeddings:
                    encoder_hidden_states = batch["input_ids"]
                else:
                    encoder_hidden_states = encode_prompt(
                        text_encoder,
                        batch["input_ids"],
                        batch["attention_mask"],
                        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                    )

                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None


                # # Add this right before your failing unet call
                # print("=== DEBUG INFO ===")
                # print(f"noisy_model_input shape: {noisy_model_input.shape}")
                # print(f"timesteps shape: {timesteps.shape}")
                # print(f"encoder_hidden_states shape: {encoder_hidden_states.shape}")
                # if class_labels is not None:
                #     print(f"class_labels shape: {class_labels.shape}")
                # print(f"UNet input channels: {unet.config.in_channels}")
                # print(f"UNet sample size: {unet.config.sample_size}")
                # print("==================")


                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                ).sample

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                    
                    # print(f'loss: {loss} prior loss: {prior_loss}')
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters)
                        if args.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                
                
                if args.placeholder_token is not None:
                    # print(f'update only placeholder token embedding')
                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

                    t = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                    # print(f"placeholder weight mean: {t[placeholder_token_ids[0]].mean()}")
                    # print(f"idx0 weight mean: {t[0].mean()}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path) # lora also saved here
                        
                        save_lora(
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder) if args.train_text_encoder else None,
                            output_dir=os.path.join(save_path)
                        )
                                
                        logger.info(f"Saved state to {save_path}")
                        
                        if args.placeholder_token is not None:
                            # ─────────────────────── save textual-inversion embeddings ───────────────────────
                            # Accept either a single token or a comma-separated list
                            placeholder_tokens = [t.strip() for t in args.placeholder_token.split(",") if t.strip()]
                            # Convert each placeholder token to its ID (list-compatible already)
                            placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

                            save_token_embedding(
                                accelerator.unwrap_model(text_encoder),
                                placeholder_tokens,
                                placeholder_token_ids,
                                accelerator,
                                osp.join(save_path, "token_embedding.pt"),
                            )
                            print(f"Saved embeddings for {len(placeholder_tokens)} token(s): {placeholder_tokens}")



                    if global_step % args.validation_steps == 0:
                        log_validation(
                            unet=unet,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            gen_dtype=args.gen_dtype,
                            
                            epoch=epoch,)
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        
        # generation branch
        if args.gen_image_path is not None:
            del unet, text_encoder
            torch.cuda.empty_cache()
    
            print(f'entering image generation, save image to: {args.gen_image_path}')
            # pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype)
            pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=args.gen_dtype)
            
            
            original_pretrained_weights =  copy.deepcopy(pipeline.unet.state_dict()) if args.use_generation_phases else None
            
            if args.load_pretrained_lora_weight_path is not None and args.load_pretrained_lora_weight_path:
                print('loading LoRA into UNet ....')
                print(f'LoRA path: {args.load_pretrained_lora_weight_path}')
            
                dummy_pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, unet=pipeline.unet, text_encoder=pipeline.text_encoder, vae=pipeline.vae, revision=args.revision)
                
                lora_pretrained_weight_ = load_file(osp.join(args.load_pretrained_lora_weight_path, "pytorch_lora_weights.safetensors"))
                lora_pretrained_weight = convert_lora_weight(lora_pretrained_weight_)
                pipeline.load_lora_weights(lora_pretrained_weight)
            
            
                # dummy_pipeline.load_lora_weights(args.load_pretrained_lora_weight_path, weight_name="pytorch_lora_weights.safetensors")

                dummy_pipeline.fuse_lora()
                print('Fused LoRA  ....')
                    
            
            if args.load_unet_weight_path is not None:
                print('loading UNet weight from: ', args.load_unet_weight_path)
                if '.safetensor' in args.load_unet_weight_path:
                    
                    
                    
                    pipeline.unet.load_state_dict(load_file(args.load_unet_weight_path), strict=False)
                    
                    
                    # this is a whole UNet parameters
                    # unlearned_weights =  copy.deepcopy(pipeline.unet.state_dict()) if args.use_generation_phases else None # None .... by default
                    
                    unlearned_weights = load_file(args.load_unet_weight_path)
                    
                    
                    # print(100*['im here'])
                else:
                    pipeline.unet.load_state_dict(torch.load(args.load_unet_weight_path), strict=False)
                print('UNet weight loaded (for generation)')
                
                
                pipeline.unet.eval()
            else:
                print('not loading UNet weight')            
             
            
            if args.load_lora_weight_path is not None and args.lora_rank is not None and args.lora_rank > 0:
                # load attention processors
                print(25*"#")
                print('loading LoRA weight')
                
                lora_pretrained_weight_ = load_file(osp.join(args.load_lora_weight_path, "pytorch_lora_weights.safetensors"))
                lora_pretrained_weight = convert_lora_weight(lora_pretrained_weight_)
                pipeline.load_lora_weights(lora_pretrained_weight)

                
                # pipeline.load_lora_weights(args.load_lora_weight_path, weight_name="pytorch_lora_weights.safetensors")
                
                print(f'generating images from: lora{args.load_lora_weight_path}')
                
            else: print('not loading loRA weight')
            if args.load_token_embedding_path is not None and args.placeholder_token is not None and osp.isdir(args.load_token_embedding_path):
                file_name = f'token_embedding-{args.load_token_embedding_step}.pt' if args.load_token_embedding_step is not None else 'token_embedding.pt'
                print(f'loading token embedding from: {osp.join(args.load_token_embedding_path,file_name)}')
                load_token_embedding(pipeline.text_encoder, pipeline.tokenizer, osp.join(args.load_token_embedding_path,file_name))
            elif args.load_token_embedding_path is not None:
                print(f'loading token embedding from: {args.load_token_embedding_path}')
                load_token_embedding(pipeline.text_encoder, pipeline.tokenizer, args.load_token_embedding_path)
                
                
            
            if args.use_generation_phases:
                
                log_validation(
                    unet=pipeline.unet,
                    text_encoder=pipeline.text_encoder,
                    tokenizer=pipeline.tokenizer,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    gen_dtype=args.gen_dtype,
                    epoch=0,
                    log_label='image generation',
                    save_image_path = args.gen_image_path,
                    original_pretrained_weights=original_pretrained_weights,
                    unlearned_weights=unlearned_weights,
                    )
            
            
            else:
                log_validation(
                    unet=pipeline.unet,
                    text_encoder=pipeline.text_encoder,
                    tokenizer=pipeline.tokenizer,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    gen_dtype=args.gen_dtype,
                    epoch=0,
                    log_label='image generation',
                    save_image_path = args.gen_image_path,
                    )
                        
                        
                                                
                        
        else:
            unet = accelerator.unwrap_model(unet)
            unet = unet.to(torch.float32)
            unet_lora_layers = unet_lora_state_dict(unet)

            if text_encoder is not None and args.train_text_encoder:
                text_encoder = accelerator.unwrap_model(text_encoder)
                text_encoder = text_encoder.to(torch.float32)
                text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder)
            else:
                text_encoder_lora_layers = None

            # /home/nessessence/anaconda3/envs/mace/lib/python3.10/site-packages/diffusers/loaders.py
            
            # Final save
            
            # save lora
            if args.lora_rank is not None and args.lora_rank > 0:
                LoraLoaderMixin.save_lora_weights(
                    save_directory=args.output_dir,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                )
            if args.placeholder_token is not None:
                # save token
                # placeholder_tokens = [args.placeholder_token]
                # placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens) # also
                # save_token_embedding(accelerator.unwrap_model(text_encoder), placeholder_tokens, placeholder_token_ids, accelerator, osp.join(args.output_dir,'token_embedding.pt'))
                            
                # ─────────────────────── save textual-inversion embeddings ───────────────────────
                # Accept either a single token or a comma-separated list
                placeholder_tokens = [t.strip() for t in args.placeholder_token.split(",") if t.strip()]

                # Convert each placeholder token to its ID (list-compatible already)
                placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

                save_token_embedding(
                    accelerator.unwrap_model(text_encoder),
                    placeholder_tokens,
                    placeholder_token_ids,
                    accelerator,
                    osp.join(args.output_dir, "token_embedding.pt"),
                )
                print(f"Saved embeddings for {len(placeholder_tokens)} token(s): {placeholder_tokens}")

            # Final inference
            # Load previous pipeline
            
            pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype)
            if args.lora_rank is not None and args.lora_rank > 0:
                # load attention processors
                print('loading LoRA weight')
                pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")
            if args.placeholder_token is not None:
                print('loading token embedding')
                load_token_embedding(pipeline.text_encoder, pipeline.tokenizer, osp.join(args.output_dir,'token_embedding.pt'))
            

            # log_validation(
            #     unet=pipeline.unet,
            #     text_encoder=pipeline.text_encoder,
            #     tokenizer=pipeline.tokenizer,
            #     args=args,
            #     accelerator=accelerator,
            #     weight_dtype=weight_dtype,
            #     gen_dtype=args.gen_dtype,
            #     epoch=epoch,
            #     log_label='final')
                        
                        

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    args.pc_id = os.environ.get("pc_id")


    if args.donot_reinit_validation_generator:
        args.reinit_validation_generator = False # used for COCO for example --> more noise
    else:
        args.reinit_validation_generator = True # by default


    ## prompt validation
    args.validation_prompt = args.validation_prompt.split(';')
    print(f'validation prompt: {args.validation_prompt}')
    
    
        # my add
    # if ',' in args.instance_prompt:
    # print(f'instance prompt: {args.instance_prompt}')
    # print(f'instance_data_dir: {args.instance_data_dir}')
    if ',' in args.instance_prompt:
        args.instance_prompt = args.instance_prompt.split(',')
        print(f'instance prompt: {args.instance_prompt}')
    if ',' in args.instance_data_dir:
        args.instance_data_dir = args.instance_data_dir.split(',')
        print(f'instance_data_dir: {args.instance_data_dir}')
        
    if args.instance_prompt[:4] == '*Ph.':
        args.use_generation_phases = True
        print('using generation phases for instance prompt')
    else:
        args.use_generation_phases = False
    
    
    args.concat_prompt_indiv = {}
    args.concat_prompt_indiv['all'] =  args.validation_prompt

    
    if args.gen_dtype == 'fp16': args.gen_dtype = torch.float16
    if args.gen_dtype == 'fp32': args.gen_dtype = torch.float32
    if args.gen_dtype == 'bf16': args.gen_dtype = torch.bfloat16
    
    if args.load_lora_weight_path == '': args.load_lora_weight_path = None
    if args.load_unet_weight_path == '': args.load_unet_weight_path = None

    # hack for automatic gen_image_path
    if args.gen_image_path is not None and args.gen_image_path=='auto':
        # Get the last two components of the path
        
        # not much reliableq
        if args.load_unet_weight_path is not None:
            last_two = os.path.join(*args.load_unet_weight_path.strip("/").split("/")[-2:])
        
        
        if args.load_lora_weight_path is not None:
            last_two = os.path.join(*args.load_lora_weight_path.strip("/").split("/")[-2:])
        if args.load_token_embedding_path is not None:
            last_two = os.path.join(*args.load_token_embedding_path.strip("/").split("/")[-2:])

        # Build the gen_image_path
        args.gen_image_path = os.path.join("data_root/generated/study", last_two)
        os.makedirs(args.gen_image_path,exist_ok=True)
        print(f'gen image path: { args.gen_image_path}')
    
    if args.gen_image_path is not None:
        if args.load_lora_weight_path and args.wait_weight:
            lora_weight_path =  osp.join(args.load_lora_weight_path, "pytorch_lora_weights.safetensors")
            # if lora_weight_path does not exist, then wait (it is in training process) ... re-check every 10 seconds
            while not osp.exists(lora_weight_path):
                print(f'waiting for lora weight: {lora_weight_path}')
                time.sleep(10)

    args.wait_weight = True
    if args.load_unet_weight_path and args.wait_weight:
        unet_weight_path =  args.load_unet_weight_path
        # if unet_weight_path does not exist, then wait (it is in training process) ... re-check every 10 seconds
        while not osp.exists(unet_weight_path):
            print(f'waiting for unet weight: {unet_weight_path}')
            time.sleep(30)
        time.sleep(50)
    
    # for mace
    if 'mace' in args.pretrained_model_name_or_path  and args.wait_weight:
        sd_weight_path =  osp.join(args.pretrained_model_name_or_path, "unet")
        while not osp.exists(sd_weight_path):
            print(f'waiting for pretrained weight: {sd_weight_path}')
            time.sleep(60)
        time.sleep(50)
        

    main(args)
    
    