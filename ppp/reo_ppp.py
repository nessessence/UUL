from train_ppp_erase import get_parser,resolve_model_name
from train_ppp_erase import main as ppp_erase
from ti_attack_utils.dataset import TextualInversionDataset
import torch.nn.functional as F
import shutil
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from tqdm import tqdm
import math
import random
import numpy as np
from safetensors.torch import load_file
from copy import deepcopy
import os
import os.path as osp

#+++ Utility Function 
        
# my own implementation
def load_token_embedding(text_encoder, tokenizer, weight_path):
    print(f"Loading Token Embeddings from {weight_path}")
    # Load the saved token embeddings
    loaded_embeds_dict = torch.load(weight_path)
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

def save_token_embedding(text_encoder, placeholder_token, placeholder_token_id, weight_path):
    print(f"Saving Token Embeddings to {weight_path}")
    # Get the input embedding weights
    token_embeddings = text_encoder.get_input_embeddings().weight
    # Build the dictionary of token -> embedding
    learned_embeds_dict = {
        token: token_embeddings[token_id].detach().cpu()
        for token, token_id in zip(placeholder_token, placeholder_token_id)
    }
    torch.save(learned_embeds_dict, weight_path)
    

    
# we change `(STEREO) diffuser` to `pipeline`
def train_concept_inversion(
    pipeline,
    placeholder_token, 
    initializer_token, 
    train_data_dir, 
    lr, 
    device, 
    num_vectors=1, 
    max_train_steps=3000,  # Total training steps across all epochs
    resolution=512, 
    learnable_property="object",
    lr_scheduler="constant", 
    lr_warmup_steps=0, 
    scale_lr=False,  # Option to scale learning rate
    center_crop=False,
    
    iteration=None,  #. (I use it for naming the token) this inherit from STEREO ... to automatically create non-overlapping image subsets for each iteration .... in which im not sure why
    num_iterations=None,
    
    output_dir=None,
    save_steps=50,
):
    
    # Set the random seed for reproducibility
    seed = 42
    np.random.seed(seed)      # For numpy
    random.seed(seed)         # For the random module
    torch.manual_seed(seed) 

    pipeline.requires_grad = False

    for param in pipeline.text_encoder.text_model.embeddings.token_embedding.parameters():
        param.requires_grad = True

    tokenizer = pipeline.tokenizer

    # Add placeholder tokens to tokenizer
    placeholder_tokens = [placeholder_token]
    additional_tokens = [f"{placeholder_token}_{i}" for i in range(1, num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(f"Token '{placeholder_token}' already exists in tokenizer. Use a different token name.")

    # Convert initializer and placeholder tokens to IDs
    initializer_token_id = tokenizer.convert_tokens_to_ids([initializer_token])[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)


    # Resize text encoder embeddings to accommodate new tokens
    pipeline.text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize placeholder token embeddings using initializer token
    with torch.no_grad():
        token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
        ctr = 0
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            ctr += 1
        print(f"Initialized {ctr} placeholder token embeddings with '{initializer_token}' token embeddings.")

    # Save the original token embeddings
    org_token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data.clone()
    

    # Set up dataset and dataloader with specified resolution
    dataset = TextualInversionDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=resolution,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        repeats=100,
        set="train",
        learnable_property=learnable_property,
        center_crop=center_crop,
        
        
        iteration=iteration, # this inherit from STEREO ... to automatically create non-overlapping image subsets for each iteration .... in which im not sure why
        num_iterations=num_iterations
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Calculate steps per epoch and number of epochs needed to reach max_train_steps
    steps_per_epoch = len(dataloader)
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)

    # Scale learning rate if specified
    if scale_lr:
        effective_batch_size = dataloader.batch_size
        lr *= effective_batch_size  # Adjust learning rate based on batch size

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(pipeline.text_encoder.get_input_embeddings().parameters(), lr=lr)
    scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=max_train_steps)

    # Initialize a single progress bar for the entire training process
    progress_bar = tqdm(total=max_train_steps, desc="Concept Inversion Attack Progress", unit="step")
    global_step = 0

    # Training loop following the epoch and step structure
    for epoch in range(num_train_epochs):
        pipeline.text_encoder.train()
        
        for step, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break

            # Zero gradients for each batch
            optimizer.zero_grad()

            # Encode images to latent space
            latents = pipeline.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 999, (latents.shape[0],), device=latents.device)
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            # Forward pass through U-Net within finetuner context
            encoder_hidden_states = pipeline.text_encoder(batch["input_ids"].to(device)).last_hidden_state
            model_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]

            # Calculate loss and backpropagate
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            # Optimizer step and scheduler update
            optimizer.step()
            torch.cuda.empty_cache()
            scheduler.step()

            # Freeze all embeddings except for the placeholder tokens
            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=device)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False  # False indicates trainable embeddings

            # Restore the frozen embeddings
            with torch.no_grad():
                pipeline.text_encoder.get_input_embeddings().weight.data[index_no_updates] = org_token_embeds[index_no_updates]
            
            
       

            # Update progress bar and global step
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            global_step += 1
            
            
            
            if global_step % save_steps == 0:
                # save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                save_path = os.path.join(output_dir ,f"tia-{iteration}iter-{global_step}step.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_token_embedding(pipeline.text_encoder,  placeholder_tokens, placeholder_token_ids,  save_path)




            if global_step >= max_train_steps:
                break

    progress_bar.close()
    
    # placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    
    save_path = os.path.join(output_dir,f"tia-{iteration}iter-{global_step}step.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_token_embedding(pipeline.text_encoder, placeholder_tokens, placeholder_token_ids, save_path)
        
            

    # Save the text encoder state dict
    # if not (save_path == None):
    #     torch.save(pipeline.text_encoder.state_dict(), save_path)
    # else:
    #     print("Not saving the text encoder state dict as save_path is None.")

    del optimizer, scheduler, dataset, dataloader, progress_bar, global_step, steps_per_epoch, num_train_epochs, effective_batch_size, token_embeds, index_no_updates, model_pred, target, loss, batch, latents, noise, timesteps, noisy_latents, encoder_hidden_states, placeholder_tokens, additional_tokens, initializer_token_id, placeholder_token_ids, tokenizer, org_token_embeds
    torch.cuda.empty_cache()
    # pipeline.eval()
    return pipeline

#--- 




if __name__ == '__main__':
    
    parser = get_parser()
    # Add new arguments specific to a.py
    
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    
    # these are defined in ppp_erase
    # parser.add_argument("--load_unet_weight_path",type=str,default=None) # many unlearned model, UCE, ESD,  # 
    # parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda')
    
    
    # REO with TI arguments
    parser.add_argument("--total_ti_attack_iterations", type=int, default=3)
    parser.add_argument("--attack_unlearn_steps", type=int,default=None) # follow max_training_step if None
    
    # TI related arguments
    parser.add_argument("--ti_max_train_steps", type=int, default=3000, help="Maximum training steps for textual inversion") # you train for 3000 steps
    parser.add_argument("--ti_used_train_steps", type=int, default=500) # you use the 500 steps token
    parser.add_argument("--ti_save_steps", type=int, default=100) # save ti token for every 100 steps
    parser.add_argument("--ti_lr", type=float, default=5e-3, help="Learning rate for textual inversion") # STEREO named it ci_lr
    parser.add_argument("--center_crop", type=bool, required=False, help="Center crop the images during training", default=False)
    
    parser.add_argument("--train_data_dir", type=str, required=False, help="Gallery images to be used during training")
    parser.add_argument("--learnable_property", type=str, required=False, help="object/style", default="object")
    parser.add_argument("--initializer_token", type=str, required=True, help="Initializer token (OPTIONS: person/object/art)")
    
    
    parser.add_argument("--load_erased_weight_if_exist",action='store_true')
    
    
    parser.add_argument("--skip_stage1",action='store_true')
    
    
    
    args = parser.parse_args()
    
    
    
    #+++ Main_Function
    
    exp_path_dir = osp.join(args.save_path, resolve_model_name(args))
    ti_attack_path_dir = osp.join(exp_path_dir,'attack_tokens')
    
    
    
    
    attack_unlearn_steps = args.attack_unlearn_steps if args.attack_unlearn_steps is not None else args.max_training_step
    
    # so that we don't need erase for the first iteration
    if args.load_erased_weight_if_exist:
        erased_unet_path = osp.join(exp_path_dir, f'step{attack_unlearn_steps}.safetensors')
        if osp.exists(erased_unet_path):
             print(f'Found erased unet weights at {erased_unet_path}, loading it directly...')
             load_unet_weight_path = erased_unet_path
        else: load_unet_weight_path = None
    else:
        load_unet_weight_path = args.load_unet_weight_path # None by default
        
        
    
    if load_unet_weight_path is None:
        print('Start the first erasing')
        ppp_erase(args)
        load_unet_weight_path = osp.join(exp_path_dir,f'step{attack_unlearn_steps}.safetensors')
        # load_unet_weight_path = osp.join(exp_path_dir,f'step{attack_unlearn_steps}.safetensors')
    
    
    saved_tokens = {}
    for cur_ti_iter in range(args.total_ti_attack_iterations):
        if args.skip_stage1: continue
        
        print(f'starting erasing-attack iter: {cur_ti_iter}')
        
        # placeholder_token = generate_unique_placeholder_token(saved_tokens, ti_iter)
        placeholder_token = f'TIA{cur_ti_iter}'
        saved_tokens[f'{cur_ti_iter}'] = placeholder_token
        
        
        # ReInitialize the pipeline for each iteration
        print('Initializing Stable Diffusion Pipleine')
        pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype='bf16', use_safetensors=True).to(args.device)
        print(f"load_unet_weight_path: {load_unet_weight_path}")
        print(f'loading Erased UNet weights from {load_unet_weight_path} for iteration {cur_ti_iter}...')
        pipeline.unet.load_state_dict(load_file(load_unet_weight_path), strict=False)
        

        print(f"Textual Inversion Attacking with '{placeholder_token}' saved to {load_unet_weight_path}")
        pipeline = train_concept_inversion(
            pipeline=pipeline, 
            placeholder_token=placeholder_token,initializer_token=args.initializer_token, learnable_property=args.learnable_property,
            train_data_dir=args.train_data_dir, lr=args.ti_lr,  scale_lr=True, max_train_steps=args.ti_max_train_steps,
            device=args.device,center_crop=args.center_crop,
            save_steps=args.ti_save_steps,
            iteration=cur_ti_iter,  # just to name the token , and some data loading operation need it (from STEREO)
            num_iterations=args.total_ti_attack_iterations, # this inherit from STEREO ... to automatically create non-overlapping image subsets for each iteration .... in which im not sure why
            output_dir=ti_attack_path_dir)
        
        del pipeline.unet, pipeline.vae, pipeline.text_encoder, pipeline
        torch.cuda.empty_cache()
            
            
        
        args_ = deepcopy(args)
        args_.load_unet_weight_path = load_unet_weight_path
        args_.load_token_embedding_path = osp.join(ti_attack_path_dir,f"tia-{cur_ti_iter}iter-{args.ti_used_train_steps}step.pt")
        args_.erase_concept_ti = f'TIA{cur_ti_iter}' # placeholder_token
        args_.attacked_flag = f"tia-{cur_ti_iter+1}iter-{args.ti_used_train_steps}step" #  osp.join(args.save_path, model_name, 'attacked_models', f'{args.attacked_flag}_step{training_step}.safetensors')
        
        print(f'performing erasing with {args_.erase_concept_ti} token loaded from {args_.load_token_embedding_path} ')
        ppp_erase(args_)
        print(f"Complete Attacking model with '{placeholder_token}' saved to {load_unet_weight_path}")
        
        
        # remove old weight (saved memory)
        if cur_ti_iter > 0: 
            os.remove(load_unet_weight_path)
            print(f'Removed previous unet weight at {load_unet_weight_path}')
        
        load_unet_weight_path = osp.join(exp_path_dir, 'attacked_models', f'tia-{cur_ti_iter+1}iter-{args.ti_used_train_steps}step_step{args.max_training_step}.safetensors')
        print(f'Next iteration will load unet from {load_unet_weight_path}')
        

    
    
    
    # Stage 2: PPP Erase with multiple attacked tokens
    print("Stage 2: PPP Erase with multiple attacked tokens")
    print(f"Starting PPP Erase with the attacked token: TIA0;TIA1;TIA2...")
    args_ = deepcopy(args)
    
    # args_.load_token_embedding_path = f"{osp.join(ti_attack_path_dir,f'tia-iter0-{args.ti_used_train_steps}step.pt')};{osp.join(ti_attack_path_dir,f'tia-iter1-{args.ti_used_train_steps}step.pt')};{osp.join(ti_attack_path_dir,f'tia-iter2-{args.ti_used_train_steps}step.pt')}
    args_.load_token_embedding_path = ";".join(osp.join(ti_attack_path_dir, f"tia-{i}iter-{args.ti_used_train_steps}step.pt") for i in range(args.total_ti_attack_iterations))

    args_.erase_concept_ti =  ';'.join(f'TIA{i}' for i in range(args.total_ti_attack_iterations)) #f'TIA0;TIA1;TIA2' # should be more automated
    args_.attacked_flag = f'tia-012iter-{args.ti_used_train_steps}step' # hardcoded
    
    ppp_erase(args_)
    
    

    torch.cuda.empty_cache()

    print(f'done erasing .... at: attacked_models')
    
#---
    

