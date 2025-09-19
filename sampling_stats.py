import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse
import numpy as np
import os.path as osp

# sys.path.append('.')
from esd.utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call

from collections import defaultdict
import torch.nn.functional as F


torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# test determinism
def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute mean squared error between two tensors.
    Returns 0.0 if they are exactly identical.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return torch.mean((a - b) ** 2).item()

def dd_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: dd_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: dd_to_dict(v) for k, v in d.items()}
    return d



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

    # erase_mat = noise_pred_erase.reshape(n_batch, -1).detach().to( device="cpu")
    # general_mat = noise_pred_general.reshape(n_batch, -1).detach().to(device="cpu")
    
    
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




exp_option = 'infer_zt' # 'infer_zt'
stored_z_t_path = "data_root/cache/compositional_latents/zt_nT50.n50.bs10.seed999_a photo of person.pt"
num_inference_steps = 50
n_samples = 50


device = 'cuda:3'
torch_dtype = torch.bfloat16


root_dir = "data_root/cache/compositional_latents"
height=width=512 
batchsize = 10 # inference batch size
guidance_scale = 3.0
timestep_cond = None

max_n_sample = 10000
root_seed = 999
rng = np.random.RandomState(root_seed)
random_seeds = rng.choice(range(2**15), size=max_n_sample, replace=True) # max [n_concept,n_seed]

# erase_concept = "Barrack Obama"
# general_concept = "person"


# stored_z_t_path = None

pipe, base_unet, esd_unet = load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype, device=device)


pipe.set_progress_bar_config(disable=True)
pipe.scheduler.set_timesteps(num_inference_steps)




seen_persons =['Barrack Obama','Rihanna','Ed Sheeran','Margot Robbie','Chris Hemsworth','Chris Evans','Amy Adams','Anne Hathaway','Mariah Carey','Octavia Spencer','Morgan Freeman','Drake']
# erase_concept = 'a photo of Barrack Obama'
# general_concept = "a photo of person"
general_concepts = ["a photo of person"]
# for erase_concept in ['a photo of Barrack Obama','a photo of Rihanna', "a photo of Anne Hathaway",  "a photo of cat", "a photo of car", "a photo of tree"]:
erase_concepts = [f'a photo of {name}' for name in seen_persons]
# erase_concepts = [f'a photo of {name}' for name in ['cat','car','tree']]

if exp_option == 'compute_zt':
    assert stored_z_t_path is  None, "Please NOT store z_t path"
    erase_concepts = len(general_concepts)*['dummy'] # only need to compute once for general concept
    print("Computing and storing z_t for general concept only:", general_concepts)
    
elif exp_option == 'infer_zt':
    general_concepts = len(erase_concepts)*general_concepts # need to infer for each erase concept

for general_concept, erase_concept in zip(general_concepts, erase_concepts):
    
    if exp_option == 'compute_zt':
        print(f"Experiment: Computing z_t for general concept '{general_concept}'")
    elif exp_option == 'infer_zt':
        print(f"Experiment: Inferring z_t for erase concept '{erase_concept}' and general concept '{general_concept}'")
    
    # computing text embeddings
    with torch.no_grad():
            erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept, device=device,num_images_per_prompt=batchsize,do_classifier_free_guidance=True,negative_prompt='')
            erase_embeds = erase_embeds.to(device)
            null_embeds = null_embeds.to(device)
            
            general_embeds, _ = pipe.encode_prompt(prompt=general_concept,device=device,num_images_per_prompt=batchsize,do_classifier_free_guidance=True,negative_prompt='')
            base_embeds = general_embeds.to(device)
            

    output_stats = defaultdict(lambda: defaultdict(list))
    stored_z_t = defaultdict(lambda: defaultdict(list))
    n_iter = int(n_samples // batchsize)

    if stored_z_t_path is not None and os.path.exists(stored_z_t_path):
        print(f"Loading existing stored_z_t from {stored_z_t_path}")
        stored_z_t = torch.load(stored_z_t_path)

    total_n =  n_samples * num_inference_steps
    with tqdm(total=total_n, desc="Processing") as pbar:
        with torch.no_grad():
            for i in range(n_iter):
                seed = int(random_seeds[i]) # downside is that this is seed per batch, not per sample
                # start with the lower timesteps (slower)
                for t_inferstep in tqdm(list(range(0, num_inference_steps-1))[::-1]):
                    t = pipe.scheduler.timesteps[t_inferstep]

                    
                    # use precomputed z_t if available
                    if stored_z_t_path is not None:
                        z_t = stored_z_t[general_concept][t.item()][i*batchsize:(i+1)*batchsize]
                        z_t = torch.stack(z_t, dim=0).to(device) # [n_batch, 4, 64, 64]
                        # print(z_t.shape)
                    else:
                        z_t = pipe(general_concept,
                                num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                run_till_timestep=t_inferstep, generator=torch.Generator().manual_seed(seed), output_type='latent', height=height, width=width).images
                        # store z_t
                        if exp_option == 'compute_zt':
                            stored_z_t[general_concept][t.item()] += [z_t[i].detach().cpu() for i in range(z_t.shape[0])]
                    
                    
                    
                    if exp_option == 'infer_zt':
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
                        
                        update_output_stats(noise_pred_erase, noise_pred_general,timestep = t.item(), erase_concept=erase_concept, general_concept=general_concept)
                        

                pbar.update(1)
                
        
        if exp_option == 'infer_zt':
            save_path = osp.join(root_dir, f'{general_concept}.{erase_concept}_zt_nT{num_inference_steps}.n{n_samples}.bs{batchsize}.seed{root_seed}_{general_concept}.pt')
            torch.save(dd_to_dict(output_stats), save_path)
            print(f"Saved output_stats to {save_path}")
        elif exp_option == 'compute_zt':
            save_path = osp.join(root_dir, f'zt_nT{num_inference_steps}.n{n_samples}.bs{batchsize}.seed{root_seed}_{general_concept}.pt')
            torch.save(dd_to_dict(stored_z_t), save_path)
            print(f"Saved stored_z_t to {save_path}")
