import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse

# sys.path.append('.')
from esd.utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

from collections import defaultdict
import torch.nn.functional as F

def nested_list():
    return defaultdict(list)



def load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet



def update_output_stats(noise_pred_erase, noise_pred_general, timestep,
                        erase_concept="erase", general_concept="general"):
    global output_stats

    # Expect [n_batch, 4, H, W]
    assert noise_pred_erase.shape == noise_pred_general.shape, \
        f"Shape mismatch: {noise_pred_erase.shape} vs {noise_pred_general.shape}"
    n_batch = noise_pred_erase.shape[0]

    # Flatten to [n_batch, D], compute in float64 on CPU for numerical stability
    erase_mat = noise_pred_erase.reshape(n_batch, -1).detach().to(dtype=torch.float64, device="cpu")
    general_mat = noise_pred_general.reshape(n_batch, -1).detach().to(dtype=torch.float64, device="cpu")

    # Per-sample norms
    norm_erase = torch.linalg.vector_norm(erase_mat, dim=1)       # [n_batch]
    norm_general = torch.linalg.vector_norm(general_mat, dim=1)   # [n_batch]

    # Pairwise dot (i with i)
    dots = (erase_mat * general_mat).sum(dim=1)                   # [n_batch]

    # Pairwise cosine (stable + clamped)
    cosines = F.cosine_similarity(erase_mat, general_mat, dim=1, eps=1e-12)  # [n_batch]
    cosines = torch.clamp(cosines, -1.0, 1.0)

    # --- Store ---
    # Raw: store each sample's flattened tensor
    output_stats[f"raw_{erase_concept}"][timestep].extend([erase_mat[i] for i in range(n_batch)])
    output_stats[f"raw_{general_concept}"][timestep].extend([general_mat[i] for i in range(n_batch)])

    # Scalars: extend lists with n_batch new values
    output_stats[f"norm_{erase_concept}"][timestep].extend(norm_erase.tolist())
    output_stats[f"norm_{general_concept}"][timestep].extend(norm_general.tolist())
    output_stats[f"dot_{erase_concept}.{general_concept}"][timestep].extend(dots.tolist())
    output_stats[f"cosine_{erase_concept}.{general_concept}"][timestep].extend(cosines.tolist())


# def update_output_stats(noise_pred_erase, noise_pred_general, timestep,
#                         erase_concept="erase", general_concept="general"):
#     global output_stats

#     # Flatten to vectors
#     erase_flat = noise_pred_erase.flatten(start_dim=0).detach().cpu()
#     general_flat = noise_pred_general.flatten(start_dim=0).detach().cpu()

#     # Norms
#     norm_erase = torch.norm(erase_flat).item()
#     norm_general = torch.norm(general_flat).item()

#     # Dot product
#     dot_val = torch.dot(erase_flat, general_flat).item()

#     # Cosine similarity
#     cosine_val = dot_val / (norm_erase * norm_general + 1e-8)

#     # Store results grouped by timestep under each key
#     output_stats[f"raw_{erase_concept}"][timestep].append(erase_flat)
#     output_stats[f"raw_{general_concept}"][timestep].append(general_flat)
#     output_stats[f"norm_{erase_concept}"][timestep].append(norm_erase)
#     output_stats[f"norm_{general_concept}"][timestep].append(norm_general)
#     output_stats[f"dot_{erase_concept}.{general_concept}"][timestep].append(dot_val)
#     output_stats[f"cosine_{erase_concept}.{general_concept}"][timestep].append(cosine_val)
    
    


device = 'cuda:1'
torch_dtype = torch.bfloat16

num_inference_steps = 50
height=width=512 
batchsize = 50
guidance_scale = 3.0
timestep_cond = None

seed = random.randint(0, 2**15)

erase_concept = "Barrack Obama"
general_concept = "person"
pipe, base_unet, esd_unet = load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype, device=device)


pipe.set_progress_bar_config(disable=True)
pipe.scheduler.set_timesteps(num_inference_steps)





with torch.no_grad():
    
        erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batchsize,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')
                                                 
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        general_embeds, _ = pipe.encode_prompt(prompt=general_concept,
                                                device=device,
                                                num_images_per_prompt=batchsize,
                                                do_classifier_free_guidance=True,
                                                negative_prompt='')
        base_embeds = general_embeds.to(device)
        
        
        

# declare once outside the function
output_stats = defaultdict(nested_list)
n_samples = 50


n_iter = int(n_samples // batchsize)


total_n =  n_samples * num_inference_steps
with tqdm(total=total_n, desc="Processing") as pbar:
    with torch.no_grad():
        for _ in range(n_iter):
            for t_inferstep in tqdm(list(range(0, num_inference_steps-1))[::-1]):
                t = pipe.scheduler.timesteps[t_inferstep]
                
                # print("timestep:", t_inferstep, t)
            
                # random t

                # print(f"t: {t}, t_inferstep: {t_inferstep}")

                # sample z_t with Pg ("person")
                z_t = pipe(general_concept, 
                        num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, 
                        run_till_timestep=t_inferstep, generator=torch.Generator().manual_seed(seed), output_type='latent', height=height, width=width).images

                # 
                noise_pred_erase = pipe.unet(
                    z_t,
                    t,
                    encoder_hidden_states=erase_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]



                noise_pred_general = pipe.unet(
                    z_t,
                    t,
                    encoder_hidden_states=general_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                
                
                # print(noise_pred_erase.shape, noise_pred_general.shape) # [10, 4, 64, 64] for 512x512 :::  [10, 4, 128, 128]
                
                # print(noise_pred_erase.shape, noise_pred_general.shape) # [4, 128, 128], [4, 128, 128]

                update_output_stats(noise_pred_erase, noise_pred_general,timestep = t, erase_concept=erase_concept, general_concept=general_concept)

                # print(output_stats)
                
            pbar.update(1)
            
            
            if t_inferstep % 10 == 0:
                torch.save(output_stats, f'data_root/cache/tmp/sampling_stats_nT{num_inference_steps}.n{n_samples}_obama_person.pth')
            
    # print(z_t.shape) # [1, 4, 64, 64] for 512x512 :::  [1, 4, 128, 128]
    
    
    torch.save(output_stats, f'data_root/cache/tmp/sampling_stats_nT{num_inference_steps}.n{n_samples}_obama_person.pth')
    