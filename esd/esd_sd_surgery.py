import os 
import os.path as osp
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import defaultdict
import torch
import random
import numpy as np

# seed = 123
# rng = np.random.RandomState(seed=seed)

    
import sys
# os.environ["PYTHONHASHSEED"] = str(123)s
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import argparse
sys.path.append('.')
from utils.sd_utils import esd_sd_call
StableDiffusionPipeline.__call__ = esd_sd_call


from gradient_surgery import collect_param_grads,zero_param_grads,param_grad_stats,generalize_gradient_projection,inject_resolved_grads_by_name
from gradient_surgery import do_grad_injection
from gradient_surgery import generalize_gradient_projection_prob
from diffusers import DDIMScheduler


concept2shortname = {
    "Margot Robbie": "mrobbie",
    "mickey mouse": "mmouse",
    "pad thai": "padthai"
}

def resolve_model_name(args): #, training_step):
    erase_concept = args.erase_concept
    train_method = args.train_method
    erase_concept_shortname = concept2shortname[erase_concept] if erase_concept in concept2shortname else erase_concept.replace(' ', '-')
    # base_file_name = f"{train_method}.{erase_concept_shortname}"


    base_file_name = f"{train_method}"
    if args.negative_guidance:
        base_file_name += f".nG{args.negative_guidance:.2f}"
    
    if not args.apply_gradient_projection and args.preservation_weight is not None and args.preservation_weight > 0:
        base_file_name += '.'
        if  args.preservation_train_set:
            if args.preservation_train_set in ['00','01','02','03']:
                base_file_name += f"pe{args.preservation_train_set}"
            elif args.preservation_train_set == 'celeb':
                base_file_name += f"cl"
            elif args.preservation_train_set == 'coco':
                base_file_name += f"cc"
            base_file_name += '-'
        
        
        if args.preservation_weight_option == 'convex' and args.preservation_weight != 0.0:
            base_file_name += f"cPS{args.preservation_weight:.2f}"
        else:
            base_file_name += f"PS{args.preservation_weight:.2f}"
        
        
    if args.apply_gradient_projection:
            base_file_name += f"_GP"
            base_file_name += '.g'
            if args.gradient_projection_param_group == 'base':
                base_file_name += f"B"
            if args.gradient_projection_param_group == 'global':
                base_file_name += f"G"
            if args.gradient_projection_param_group == 'attn_head':
                base_file_name += f"H"
            if args.gradient_projection_param_group == 'layer':
                base_file_name += f"L"
            if args.gradient_projection_param_group == 'neuron':
                base_file_name += f"N"
                
            base_file_name += '.p'
            if args.gradient_projection_mode == 'hard':
                base_file_name += f"H"
            if args.gradient_projection_mode == 'soft':
                base_file_name += f"S"
                
            if args.unlearn_proj_prob < 1.0:
                base_file_name += f"-u{args.unlearn_proj_prob:.2f}"
                
            base_file_name += '.'
            if  args.preservation_train_set:
                if args.preservation_train_set in ['00','01','02','03']:
                    base_file_name += f"pe{args.preservation_train_set}"
                elif args.preservation_train_set == 'celeb':
                    base_file_name += f"cl"
                elif args.preservation_train_set == 'coco':
                    base_file_name += f"cc"
                base_file_name += '-'

            if args.gradient_projection_preserve_scale is not None :
                
                if args.preservation_weight_option == 'convex' and args.gradient_projection_preserve_scale != 0.0 :
                    base_file_name += f"cPS{args.gradient_projection_preserve_scale:.2f}"
                else:
                    base_file_name += f"PS{args.gradient_projection_preserve_scale:.2f}"
                

                
            # if args.preservation_weight is not None and args.preservation_weight > 0:
            #     base_file_name += f"PS{args.preservation_weight:.2f}"


    if args.timestep_constraint is not None:
        base_file_name += f"_T{args.timestep_constraint}"
        
    if args.base_concept == 'general':
        base_file_name += f"_BGeneral"
        
    if args.decompositional_timestep_sampler is not None:
        base_file_name += f"_dT{args.decompositional_timestep_sampler}"

    base_file_name += f"_U.{erase_concept_shortname}"
    base_file_name += "_sd1.4"  
    base_file_name += f"_r{args.seed}"
    
    
    
    return f"{base_file_name}"


def save_esd_model(esd_param_names, esd_params, args, training_step, total_grad_stats=None):
    model_name = resolve_model_name(args)

    save_model_path = osp.join(args.save_path, model_name, f'step{training_step}.safetensors')
    os.makedirs(osp.dirname(save_model_path), exist_ok=True)

    esd_param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        esd_param_dict[name] = param
    save_file(esd_param_dict, save_model_path)
    print(f"ESD model saved at: {save_model_path}")
    
    
    if total_grad_stats is not None and len(total_grad_stats) > 0:
        grad_stats_path = osp.join(args.save_path, model_name, f'step{training_step}_grad_stats_{args.collect_gradient_statistics_option}.pt')
        os.makedirs(osp.dirname(grad_stats_path), exist_ok=True)
        torch.save(total_grad_stats, grad_stats_path)
        print(f"Gradient statistics saved at: {grad_stats_path}")
        
        # reset
        # total_grad_stats = defaultdict(list)


        


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
    parser.add_argument('--max_training_step', help='Number of max_training_step', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=2)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sd/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    


    parser.add_argument('--timestep_constraint', help='timestep constraint for diffusion model', type=str, required=False, default=None)
    parser.add_argument('--base_concept', type=str, choices=['null','general','erased'], default='null', required=False)
    
    
    parser.add_argument('--preservation_weight', type=float,  default=None, required=False)
    parser.add_argument('--preservation_train_set', type=str,  default='celeb', choices=['celeb','coco'] + ['00','01','02','03'])
    parser.add_argument('--preservation_weight_option', type=str,  default='additive', choices=['additive','convex'])



    parser.add_argument('--decompositional_timestep_sampler',  type=str,  choices=[None,'avg','indiv'], default=None)

    parser.add_argument('--apply_gradient_projection',  action='store_true', default=False)
    parser.add_argument('--gradient_projection_mode', type=str, choices=['hard','soft','none'], default='hard')
    parser.add_argument('--gradient_projection_param_group', type=str, choices=['base','global','attn_head','layer','neuron'], default='attn_head')
    parser.add_argument('--gradient_projection_preserve_scale', type=float,  default=1.0)
    
    
    
    parser.add_argument('--seed', type=int,  default=123)
    parser.add_argument('--train_precision', type=str,  default='fp32', choices=['bf16','fp32'])
    parser.add_argument('--log_step', type=int,  default=100)
    
    
    parser.add_argument('--unlearn_proj_prob', type=float,  default=1.00)

    parser.add_argument('--collect_gradient_statistics_option', type=str,  default=None, choices=[None, 'none','static', 'dynamic'])



    args = parser.parse_args()
    
    
    
    print(f'random seed: {args.seed}')
    rng = np.random.RandomState(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True # tested, does not increase training time
    torch.backends.cudnn.benchmark = False
    
    
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    # if args.unlearn_proj_prob < 1.0:
    proj_rng = torch.Generator(device=args.device) 
    proj_rng.manual_seed(args.seed **2)

    total_grad_stats = defaultdict(list)
    if args.collect_gradient_statistics_option is not None and args.collect_gradient_statistics_option in ['dynamic','static']:
        print(f"Only collecting gradient statistics: {args.collect_gradient_statistics_option}")
        
        
        # total_grad_stats['cosine_similarities'] = []
        # total_grad_stats['norms_A'] = []    
        # total_grad_stats['norms_B'] = []    


    erase_concept = args.erase_concept

    num_inference_steps = args.num_inference_steps
    
    guidance_scale = args.guidance_scale
    negative_guidance = args.negative_guidance
    train_method=args.train_method
    max_training_step = args.max_training_step
    batchsize = 1
    # height=width=1024 # Fix to 1024 ?
    height=width=512 # I now fixed this to 512
    lr = args.lr
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    
    
    criteria = torch.nn.MSELoss()
    # torch_dtype = torch.bfloat16 : will underflow, not stable
    if args.train_precision == 'bf16':
        torch_dtype = torch.bfloat16
    elif args.train_precision == 'fp32':
        torch_dtype = torch.float32 # double training time compared to bf16


    pipe, base_unet, esd_unet = load_sd_models(basemodel_id="CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.disable_xformers_memory_efficient_attention()

    base_unet = base_unet.eval()
    

    esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    # print(esd_param_names)


    # my add
    if args.preservation_train_set == 'celeb':
        preservation_concepts =  torch.load('../data_root/cache/celeb/100celebrity.pt')
        
        
    elif args.preservation_train_set == '00':
        preservation_concepts =  torch.load('../data_root/data/preservation_concepts/all_pe_v1_r123.pth')[args.erase_concept.lower()]['train']['Strongly Associated']
        print(f"preservation_concepts: {preservation_concepts}")

    if args.apply_gradient_projection:

        unet = esd_unet  ## or pipe.unet, depending on your context
        learnable_param_names, learnable_params = esd_param_names, esd_params


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
        

    
    
    pbar = tqdm(range(max_training_step+1), desc='Training ESD')
    losses = []
    for training_step in pbar:
        optimizer.zero_grad()
        
        if training_step % args.log_step == 0:
            save_esd_model(esd_param_names, esd_params, args, training_step, total_grad_stats)

            # reset
            total_grad_stats = defaultdict(list)
            
        # get the noise predictions for erase concept
        pipe.unet = base_unet
        

        if args.decompositional_timestep_sampler == 'avg':
            timestep =  rng.choice(sampler_stats['timesteps'], p=sampler_stats['probs'])
            num_inference_step_ = timesteps2num_inference_step[timestep]
            print(f"timestep: {timestep} - num_inference_step_: {num_inference_step_}")
            timestep = torch.tensor(timestep).to(device)
        elif args.timestep_constraint is not None:

            num_inference_step_ = rng.randint(0, len(constrainted_timesteps)-1)
            timestep = constrainted_timesteps[num_inference_step_]
            print(f"timestep: {timestep}")
            
            # timestep = pipe.scheduler.timesteps[run_till_timestep]
            
            # print(f"timestep: {timestep}") 
            # print(pipe.scheduler.timesteps) # reverse order : 981-1
            # print(f'effective timestep: {pipe.scheduler.timesteps[args.scaled_lb_timestep_constraint]} - {pipe.scheduler.timesteps[args.scaled_ub_timestep_constraint-1]}')

        else:
            num_inference_step_ = rng.randint(0, num_inference_steps-1)
            timestep = pipe.scheduler.timesteps[num_inference_step_] # [981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,161, 141, 121, 101,  81,  61,  41,  21,   1]
            
            
        forward_seed = rng.randint(0, 2**15)
        
        # pretrained prediction
        with torch.no_grad():
            # sample xt with Pe (reverse process)
            xt = pipe(erase_concept , 
                      num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images

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

                preservation_concept = rng.choice(preservation_concepts).item()
                print(f"preservation_concept: {preservation_concept}")
                preservation_embeds, _ = pipe.encode_prompt(prompt=preservation_concept,
                                                            device=device,
                                                            num_images_per_prompt=batchsize,
                                                            do_classifier_free_guidance=True,
                                                            negative_prompt='')

                preservation_embeds = preservation_embeds.to(device)    
                                                            
                xt_ps = pipe(preservation_concept, 
                        num_images_per_prompt=batchsize, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, run_till_timestep=num_inference_step_, generator=torch.Generator().manual_seed(forward_seed), output_type='latent', height=height, width=width).images
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
        
        
        optimizer.zero_grad(set_to_none=True)

        if (not args.apply_gradient_projection):
            # ---- baseline (same as before) ----
            
            if args.preservation_weight :
                total_loss = unlearn_loss + args.preservation_weight * preservation_loss
            else:
                total_loss = unlearn_loss
            print(f"total_loss: {total_loss.item()}, "
                f"unlearn_loss: {unlearn_loss.item()}, "
                f"preservation_loss: {preservation_loss.item()}")
            total_loss.backward()
            optimizer.step()

        else:
            # print(unlearn_loss, preservation_loss)
            print('unlearn_loss:', unlearn_loss.item(), 'preservation_loss:', preservation_loss.item())
            # ---- gradient surgery: resolve conflicts by projecting UNLEARN ⟂ PRESERVE ----
            
            # for layer_name,grad in unlearn_param_grads.items():
            # print("unlearn grad ", layer_name, grad.shape)
            
            unlearn_slice = slice(0, (2*batchsize)//2)
            preservation_slice = slice((2*batchsize)//2, 2*batchsize)
            
            # A) backprop Unlearning Loss
            unlearn_loss.backward(retain_graph=True)
            unlearn_param_grads = collect_param_grads(unet, learnable_param_names) 
            
            # set zero grads before next backprop
            # print(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.weight.grad)
            zero_param_grads(unet, learnable_param_names, set_to_none=False)
            # print(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.weight.grad)

            # B) backprop Preservation Loss
            preservation_loss.backward(retain_graph=True)
            preserve_param_grads = collect_param_grads(unet, learnable_param_names) 
            
            # C) resolve conflicts (do projection and combine with preservation gradient)
            
            if args.unlearn_proj_prob < 1.0 or (args.collect_gradient_statistics_option is not None and args.collect_gradient_statistics_option in ['dynamic','static']):
                
                if args.collect_gradient_statistics_option is not None and args.collect_gradient_statistics_option in ['dynamic','static']:
                    resolved_grads, grad_stats = generalize_gradient_projection_prob(
                        unlearn_param_grads, preserve_param_grads,
                        param_group_type=args.gradient_projection_param_group,                      # 'global' | 'attn_head'
                        projection_mode=args.gradient_projection_mode,     # 'hard' | 'soft'
                        scale=args.gradient_projection_preserve_scale,     # weight for PRESERVE branch
                        preservation_weight_option=args.preservation_weight_option,
                        A_proj_prob=args.unlearn_proj_prob,
                        rng=proj_rng,
                        collect_statistics=True
                    )
                    
                    for key in grad_stats:
                        total_grad_stats[key] += [grad_stats[key].detach().cpu()]


                        print(f"total {key}: {len(total_grad_stats[key])}")
                        
                    total_grad_stats['timesteps'].append(timestep.detach().cpu().item())
                    
                else:
                    resolved_grads = generalize_gradient_projection_prob(
                        unlearn_param_grads, preserve_param_grads,
                        param_group_type=args.gradient_projection_param_group,                      # 'global' | 'attn_head'
                        projection_mode=args.gradient_projection_mode,     # 'hard' | 'soft'
                        scale=args.gradient_projection_preserve_scale,     # weight for PRESERVE branch
                        preservation_weight_option=args.preservation_weight_option,
                        A_proj_prob=args.unlearn_proj_prob,
                        rng=proj_rng
                    )

            
            else:
                resolved_grads = generalize_gradient_projection(
                    unlearn_param_grads, preserve_param_grads,
                    param_group_type=args.gradient_projection_param_group,                      # 'global' | 'attn_head'
                    projection_mode=args.gradient_projection_mode,     # 'hard' | 'soft'
                    scale=args.gradient_projection_preserve_scale,     # weight for PRESERVE branch
                    preservation_weight_option=args.preservation_weight_option
                )

            # D) inject and step
            unet.zero_grad(set_to_none=True)
            
            # do not update if only collecting gradient statistics: static option
            if args.collect_gradient_statistics_option is not None and args.collect_gradient_statistics_option == 'static': 
                print("Skipping gradient injection/step since only collecting static gradient statistics.")
                continue
            
            do_grad_injection(unet,resolved_grads,show_per_param=False)
            optimizer.step()


        # save_esd_model(esd_param_names, esd_params, args, args.training_step)
        

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


    
    # esd_param_dict = {}
    # for name, param in zip(esd_param_names, esd_params):
    #     esd_param_dict[name] = param
    
    
    
    # Resolve naming
    # erase_concept_from = erase_concept
    # base_file_name = f"esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}-{train_method.replace('-','')}"
    
    # esd-x.obama_sd1.4
    # erase_concept_shortname = concept2shortname[erase_concept] if erase_concept in concept2shortname else erase_concept.replace(' ', '-')
    # base_file_name = f"{train_method}.{erase_concept_shortname}"

    # if args.negative_guidance != 2:
    #     base_file_name += f"_nG{args.negative_guidance:.2f}"
    
    # if not args.gradient_projection_preserve_scale and args.preservation_weight is not None and args.preservation_weight > 0:
    #     base_file_name += f"_PS{args.preservation_weight:.2f}"
        
        
    # if args.apply_gradient_projection:
    #         base_file_name += f"_GP"
            
    #         base_file_name += '.g'
    #         if args.gradient_projection_param_group == 'global' or args.gradient_projection_param_group == 'none':
    #             base_file_name += f"G"
    #         if args.gradient_projection_param_group == 'attn_head':
    #             base_file_name += f"H"
    #         if args.gradient_projection_param_group == 'layer':
    #             base_file_name += f"L"
    #         if args.gradient_projection_param_group != 'neuron':
    #             base_file_name += f"N"
                
    #         base_file_name += '.p'
    #         if args.gradient_projection_mode == 'hard':
    #             base_file_name += f"H"
    #         if args.gradient_projection_mode == 'soft':
    #             base_file_name += f"S"

    #         base_file_name += f".ps{args.gradient_projection_preserve_scale:.2f}"


    # if args.timestep_constraint is not None:
    #     base_file_name += f"_T{args.timestep_constraint}"
        
    # if args.base_concept == 'general':
    #     base_file_name += f"_BGeneral"
        
    # if args.decompositional_timestep_sampler is not None:
    #     base_file_name += f"_dT{args.decompositional_timestep_sampler}"
        
    # if args.max_training_step != 200:
    #     base_file_name += f"_S{args.max_training_step}"
        
    # base_file_name += f"_sd1.4"
    
    # save_file(esd_param_dict, f"{save_path}/{base_file_name}.safetensors")









# def set_random(seed):
#     rng = np.random.RandomState(seed=seed)
#     return rng
    # random.seed(seed)
    # np.random.seed(seed)

        
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False


    # try:
    #     torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    # except Exception:
    #     # older PyTorch
    #     torch.backends.cuda.enable_flash_sdp(False)
    #     torch.backends.cuda.enable_mem_efficient_sdp(False)
    #     torch.backends.cuda.enable_math_sdp(True)
