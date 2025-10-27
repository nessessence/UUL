import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse
import numpy as np
sys.path.append('.')
from utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call


# from gradient_surgery import AttentionGradientHook,install_hook_on_learnables,check_hook_installation,generalize_gradient_projection
from _gradient_surgery import (
    AttentionGradientHook,
    install_hook_on_learnables,
    collect_grads_by_name_from_unet,
    zero_cached_grads_in_unet,
    inject_resolved_grads_by_name,
    clear_hook_caches,
    generalize_gradient_projection,
)

# def prepare_unet_for_surgery(unet):
#     # Turn off gradient checkpointing and memory-efficient attention that can hide intermediates
#     try:
#         unet.disable_gradient_checkpointing()
#     except Exception:
#         pass
#     # If you had enabled xformers / mem-efficient attention elsewhere, prefer SDPA path:
#     try:
#         unet.set_default_attn_processor()  # reset to stock AttnProcessor2_0 first
#     except Exception:
#         pass

def prepare_unet_for_surgery(unet):
    # 1) disable gradient checkpointing (it can re-create tensors and break retain_grad capture)
    try:
        unet.disable_gradient_checkpointing()
    except Exception:
        pass

    # 2) prefer SDPA math kernels globally as a safety net
    torch.backends.cuda.matmul.allow_tf32 = True  # optional perf
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    # 3) if you had enabled xFormers somewhere, reset to stock
    try:
        unet.set_default_attn_processor()  # restores AttnProcessor2_0 path
    except Exception:
        pass

    
@torch.no_grad()
def _sanity_check_hook_flow(unet, hook_cls, verbose=True):
    """
    Checks that at least one hooked processor has a non-empty tensor in cache
    and gets a non-zero gradient after a dummy backward.
    """
    # find one processor
    one_name, one_proc = None, None
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, hook_cls):
            one_name, one_proc = name, proc
            break
    if one_proc is None:
        if verbose: print("❌ No hook instances found on UNet.attn_processors.")
        return False

    # did forward populate cache?
    if one_name not in one_proc.cache_by_name:
        if verbose: print("⚠️ Hook cache empty. Did you run a forward that hits attention?")
        return False

    # print(one_name) # down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor
    t = one_proc.cache_by_name[one_name]
    if t.grad is None or (t.grad.abs().sum().item() == 0):
        if verbose: print("⚠️ Gradient at cached tensor is zero or None.")
        return False

    if verbose: print("✅ Hook cache has non-zero grads. Flow looks good.")
    return True



def load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
    esd_params = []
    esd_param_names = []
    for name, module in esd_unet.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn2' in name:
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-u' and ('attn2' not in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-all' :
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)
                    
            if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters():
                    esd_param_names.append(name+'.'+n)
                    esd_params.append(p)

    return esd_param_names, esd_params
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SDv1.4',
                    description = 'Finetuning stable-diffusion to erase the concepts')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=3)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-u, esd-a, esd-x-strict)', type=str, required=True)
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=2)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sd/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    


    parser.add_argument('--timestep_constraint', help='timestep constraint for diffusion model', type=str, required=False, default=None)
    parser.add_argument('--base_concept', type=str, choices=['null','general','erased'], default='null', required=False)
    parser.add_argument('--preservation_weight', type=float,  default=None, required=False)



    parser.add_argument('--decompositional_timestep_sampler',  type=str,  choices=[None,'avg','indiv'], default=None)

    parser.add_argument('--apply_gradient_projection',  action='store_true', default=False)
    parser.add_argument('--gradient_projection_mode', type=str, choices=['hard','soft','none'], default='hard')
    parser.add_argument('--gradient_projection_param_group', type=str, choices=['global','attn_head','none'], default='attn_head')
    parser.add_argument('--gradient_projection_preserve_scale', type=float,  default=1.0)

    args = parser.parse_args()
    
    



    erase_concept = args.erase_concept

    num_inference_steps = args.num_inference_steps
    
    guidance_scale = args.guidance_scale
    negative_guidance = args.negative_guidance
    train_method=args.train_method
    iterations = args.iterations
    batchsize = 1
    # height=width=1024 # Fix to 1024 ?
    height=width=512 # I now fixed this to 512
    lr = args.lr
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    torch_dtype = torch.bfloat16 
    criteria = torch.nn.MSELoss()

    pipe, base_unet, esd_unet = load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)







    # my add
    preservation_concepts =  torch.load('../data_root/cache/celeb/100celebrity.pt')

    if args.apply_gradient_projection:
        # target_params = install_hook_on_learnables(esd_unet, AttentionGradientHook, learnable_param_names=esd_param_names)
        # check_hook_installation(esd_unet, AttentionGradientHook)
        
        # After you have `unet` and (optionally) your learnable_param_names list:
        
        unet = esd_unet  ## or pipe.unet, depending on your context
        # prepare_unet_for_surgery(unet)
        hooked_names = install_hook_on_learnables(
            unet,
            AttentionGradientHook,
            learnable_param_names=esd_param_names  # or your filtered learnable list
        )
        print(f"[hook] installed on {len(hooked_names)} attention processors")


    if args.timestep_constraint is not None:
        args.lb_timestep_constraint, args.ub_timestep_constraint = map(int, args.timestep_constraint.split('-'))
        print(f'timestep constraint: {args.lb_timestep_constraint}-{args.ub_timestep_constraint}')
        constrainted_timesteps = torch.tensor([ t for t in pipe.scheduler.timesteps if t < args.ub_timestep_constraint and t > args.lb_timestep_constraint ]).to(args.device)
        print(f"constrainted_timesteps: {constrainted_timesteps}")
        print(f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept.replace(' ', '_')}-{train_method.replace('-','')}_T{args.timestep_constraint}.safetensors")
        # t_scale = args.num_inference_steps/1000

        # args.scaled_lb_timestep_constraint = int(t_scale*args.lb_timestep_constraint)
        # args.scaled_ub_timestep_constraint = int(t_scale*args.ub_timestep_constraint)

        # print(f'scaled timestep constraint: {args.scaled_lb_timestep_constraint}-{args.scaled_ub_timestep_constraint}')

    if args.decompositional_timestep_sampler is not None:
        if args.decompositional_timestep_sampler == 'avg':
            sampler_general_concept = 'a photo of person'
            sampler_stats = torch.load(f'../data_root/cache/compositional_latents/minmax_inverse_cosine_AllPerson.{sampler_general_concept}_zt_nT50.n50.bs10.seed999_{sampler_general_concept}.pt')
            print('Using avg decompositional timestep sampler')
            
        timesteps2num_inference_step = {t: i for i, t in enumerate(pipe.scheduler.timesteps.tolist())}

    with torch.no_grad():
        # get prompt embeds
        erase_embeds, null_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                       device=device,
                                                       num_images_per_prompt=batchsize,
                                                       do_classifier_free_guidance=True,
                                                       negative_prompt='')
                                                 
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        
        
        if args.base_concept == 'null':
            base_embeds = null_embeds
        elif args.base_concept == 'general':
            # fix a photo of (?)
            general_embeds, _ = pipe.encode_prompt(prompt="Person",
                                                        device=device,
                                                        num_images_per_prompt=batchsize,
                                                        do_classifier_free_guidance=True,
                                                        negative_prompt='')
            base_embeds = general_embeds.to(device)
        
        
        timestep_cond = None 
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=torch_dtype)
        

    
    
    pbar = tqdm(range(iterations), desc='Training ESD')
    losses = []
    for iteration in pbar:
        optimizer.zero_grad()
        # get the noise predictions for erase concept
        pipe.unet = base_unet
        

        if args.decompositional_timestep_sampler == 'avg':
            timestep =  np.random.choice(sampler_stats['timesteps'], p=sampler_stats['probs'])
            num_inference_step_ = timesteps2num_inference_step[timestep]
            print(f"timestep: {timestep} - num_inference_step_: {num_inference_step_}")
            timestep = torch.tensor(timestep).to(device)
        elif args.timestep_constraint is not None:

            num_inference_step_ = random.randint(0, len(constrainted_timesteps)-1)
            timestep = constrainted_timesteps[num_inference_step_]
            print(f"timestep: {timestep}")
            
            # timestep = pipe.scheduler.timesteps[run_till_timestep]
            
            # print(f"timestep: {timestep}") 
            # print(pipe.scheduler.timesteps) # reverse order : 981-1
            # print(f'effective timestep: {pipe.scheduler.timesteps[args.scaled_lb_timestep_constraint]} - {pipe.scheduler.timesteps[args.scaled_ub_timestep_constraint-1]}')

        else:
            num_inference_step_ = random.randint(0, num_inference_steps-1)
            timestep = pipe.scheduler.timesteps[num_inference_step_] # [981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,161, 141, 121, 101,  81,  61,  41,  21,   1]
            
            
        seed = random.randint(0, 2**15)
        
        # pretrained prediction
        with torch.no_grad():
            # sample xt with Pe (reverse process)
            xt = pipe(erase_concept , 
                      num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(seed), output_type='latent', height=height, width=width).images

            noise_pred_erase = pipe.unet(
                xt,
                timestep,
                encoder_hidden_states=erase_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
            # get the noise predictions for null embeds
            noise_pred_base = pipe.unet(
                xt,
                timestep,
                encoder_hidden_states=base_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
   
            noise_pred_erase_from = noise_pred_erase
                
            if args.preservation_weight is not None and args.preservation_weight > 0:
        
                preservation_concept = random.choice(preservation_concepts)
                print(f"preservation_concept: {preservation_concept}")
                preservation_embeds, _ = pipe.encode_prompt(prompt=preservation_concept,
                                                            device=device,
                                                            num_images_per_prompt=batchsize,
                                                            do_classifier_free_guidance=True,
                                                            negative_prompt='')

                preservation_embeds = preservation_embeds.to(device)    
                                                            
                xt_ps = pipe(preservation_concept, 
                        num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(seed), output_type='latent', height=height, width=width).images
                noise_pred_ps = pipe.unet(
                    xt_ps,
                    timestep,
                    encoder_hidden_states=preservation_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                

        # gradient !!!
                
        pipe.unet = esd_unet
        if args.preservation_weight is not None and args.preservation_weight > 0:
            text_embeds, _ = pipe.encode_prompt(prompt=[erase_concept,preservation_concept],
                                                device=device,
                                                num_images_per_prompt=batchsize,
                                                do_classifier_free_guidance=True,
                                                negative_prompt=['',''])
            total_xt = torch.cat([xt, xt_ps], dim=0)
        else: 
            text_embeds = erase_embeds
            total_xt = xt
            
        # have to forward once
        total_noise_pred_esd_model = pipe.unet(
            total_xt,
            timestep,
            encoder_hidden_states=text_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        
        
        # print(total_noise_pred_esd_model.shape)  # [2, 4, 64, 64]
        if args.preservation_weight is not None and args.preservation_weight > 0: 
            noise_pred_esd_model, noise_pred_ps_esd_model = total_noise_pred_esd_model.chunk(2, dim=0)
            # print(noise_pred_ps_esd_model)
            # print(noise_pred_esd_model.shape, noise_pred_ps_esd_model.shape) [1, 4, 64, 64]
            unlearn_loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_base))) 
            preservation_loss = criteria(noise_pred_ps_esd_model, noise_pred_ps)
            
            
        else:
            noise_pred_esd_model = total_noise_pred_esd_model
            unlearn_loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_base))) 
            preservation_loss = torch.tensor(0.0).to(device)
        
        
        # preservation_loss = torch.tensor(0.0).to(device)
        # if args.preservation_weight is not None and args.preservation_weight > 0:
            
        #     pipe.unet = base_unet
        #     with torch.no_grad():
        #         preservation_concept = random.choice(preservation_concepts)
        #         print(f"preservation_concept: {preservation_concept}")
        #         preservation_embeds, _ = pipe.encode_prompt(prompt=preservation_concept,
        #                                                     device=device,
        #                                                     num_images_per_prompt=batchsize,
        #                                                     do_classifier_free_guidance=True,
        #                                                     negative_prompt='')

        #         preservation_embeds = preservation_embeds.to(device)    
                
                

                                                            
        #         xt_ps = pipe(preservation_concept, 
        #                 num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(seed), output_type='latent', height=height, width=width).images



        #         noise_pred_ps = pipe.unet(
        #             xt_ps,
        #             timestep,
        #             encoder_hidden_states=preservation_embeds,
        #             timestep_cond=timestep_cond,
        #             cross_attention_kwargs=None,
        #             added_cond_kwargs=None,
        #             return_dict=False,
        #         )[0]
                
        #     pipe.unet = esd_unet
        #     noise_pred_ps_esd_model = pipe.unet(
        #         xt_ps,
        #         timestep,
        #         encoder_hidden_states=preservation_embeds,
        #         timestep_cond=timestep_cond,
        #         cross_attention_kwargs=None,
        #         added_cond_kwargs=None,
        #         return_dict=False,
        #     )[0]
                    
        #     preservation_loss = criteria(noise_pred_ps_esd_model, noise_pred_ps)
            
            
            
        optimizer.zero_grad(set_to_none=True)

        if (not args.apply_gradient_projection) or (args.gradient_projection_mode == "none"):
            # ---- baseline (same as before) ----
            total_loss = unlearn_loss + args.preservation_weight * preservation_loss
            print(f"total_loss: {total_loss.item()}, "
                f"unlearn_loss: {unlearn_loss.item()}, "
                f"preservation_loss: {preservation_loss.item()}")
            total_loss.backward()
            optimizer.step()
            clear_hook_caches(unet, AttentionGradientHook)

        else:
            print(unlearn_loss, preservation_loss)
            print('unlearn_loss:', unlearn_loss.item(), 'preservation_loss:', preservation_loss.item())
            # ---- gradient surgery: resolve conflicts by projecting UNLEARN ⟂ PRESERVE ----
            
            
            
            unlearn_slice = slice(0, (2*batchsize)//2)
            preservation_slice = slice((2*batchsize)//2, 2*batchsize)
            
            # A) backward A
            unlearn_loss.backward(retain_graph=True)
            grads_A = collect_grads_by_name_from_unet(unet, AttentionGradientHook,batch_slice= unlearn_slice ) # first half = unlearn
            
            any_nonzero_A = False
            for name, proc in unet.attn_processors.items():
                if isinstance(proc, AttentionGradientHook):
                    t = proc.cache_by_name.get(name, None)    # (B*H,Q,Dh)
                    if t is not None and t.grad is not None:
                        # reshape to (B,H,Q,Dh) just to inspect slice_B
                        Btot, H, Q, Dh = proc.meta_by_name[name]
                        g_full = t.grad.view(Btot, H, Q, Dh)
                        if g_full[unlearn_slice].abs().sum().item() > 0:
                            any_nonzero_A = True
                            break
            print("[probe] nonzero attention grads for Unlearn slice? ", any_nonzero_A)

            # print(grads_A) # 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor': tensor([], device='cuda:3',size=(0, 8, 4096, 40)
            zero_cached_grads_in_unet(unet, AttentionGradientHook)
            # print(grads_A) # 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor': tensor([], device='cuda:3',size=(0, 8, 4096, 40)

            # B) backward B
            preservation_loss.backward(retain_graph=True)
            # preservation_loss.backward() # no need to retain graph here
            grads_B = collect_grads_by_name_from_unet(unet, AttentionGradientHook,batch_slice= preservation_slice ) # second half = preserve
            
            any_nonzero_B = False
            for name, proc in unet.attn_processors.items():
                if isinstance(proc, AttentionGradientHook):
                    t = proc.cache_by_name.get(name, None)    # (B*H,Q,Dh)
                    if t is not None and t.grad is not None:
                        # reshape to (B,H,Q,Dh) just to inspect slice_B
                        Btot, H, Q, Dh = proc.meta_by_name[name]
                        g_full = t.grad.view(Btot, H, Q, Dh)
                        if g_full[preservation_slice].abs().sum().item() > 0:
                            any_nonzero_B = True
                            break
            print("[probe] nonzero attention grads for Preserve slice? ", any_nonzero_B)

            
            # C) resolve conflicts and add weighted preservation gradient
            param_group = args.gradient_projection_param_group
            if param_group == "none":   # keep a sane default if 'none' was passed here
                param_group = "attn_head"

            resolved_grads = generalize_gradient_projection(
                grads_A, grads_B,
                param_group_type=param_group,                      # 'global' | 'attn_head'
                projection_mode=args.gradient_projection_mode,     # 'hard' | 'soft'
                scale=args.gradient_projection_preserve_scale,     # weight for PRESERVE branch
            )
            print('finish resolving grads')

            # D) inject and step
            unet.zero_grad(set_to_none=True)
            inject_resolved_grads_by_name(unet, AttentionGradientHook, resolved_grads)
            optimizer.step()
            clear_hook_caches(unet, AttentionGradientHook)


        # logging identical to before
        # losses.append((unlearn_loss + args.preservation_weight * preservation_loss).item())
        # pbar.set_postfix(esd_loss=losses[-1], timestep=num_inference_step_)

            
            
        # total_loss = unlearn_loss + args.preservation_weight*preservation_loss
        # print(f"total_loss: {total_loss.item()}, unlearn_loss: {unlearn_loss.item()}, preservation_loss: {preservation_loss.item()}")
        
        
        # total_loss.backward()
        # losses.append(total_loss.item())
        # pbar.set_postfix(esd_loss=total_loss.item(),
        #                  timestep=num_inference_step_,)
        # optimizer.step()
        
        # logging identical to before
        # losses.append((unlearn_loss + args.preservation_weight * preservation_loss).item())
        # pbar.set_postfix(esd_loss=losses[-1], timestep=num_inference_step_)


    
    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    if erase_concept_from is None:
        erase_concept_from = erase_concept
    
    
    base_file_name = f"esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}-{train_method.replace('-','')}"
    
    if args.negative_guidance != 2:
        base_file_name += f"_nG{args.negative_guidance:.2f}"
    
    if args.preservation_weight is not None and args.preservation_weight > 0:
        base_file_name += f"_PS{args.preservation_weight:.2f}"
    
    if args.timestep_constraint is not None:
        base_file_name += f"_T{args.timestep_constraint}"
        
    if args.base_concept == 'general':
        base_file_name += f"_BGeneral"
        
    if args.decompositional_timestep_sampler is not None:
        base_file_name += f"_dT{args.decompositional_timestep_sampler}"
        
    if args.iterations != 200:
        base_file_name += f"_step{args.iterations}"
    
    save_file(esd_param_dict, f"{save_path}/{base_file_name}.safetensors")


