import os, gc
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from omegaconf import OmegaConf
import argparse
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, DDIMScheduler
import os.path as osp
from tqdm.auto import tqdm

# my add
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
            

def main(args):

    # args.gen_dtype = 'fp16's
    if args.gen_dtype == 'fp16': gen_dtype = torch.float16
    if args.gen_dtype == 'fp32':gen_dtype = torch.float32
    
    print(f'generation dtype: {args.gen_dtype}')
    model_id = args.pretrained_model_name_or_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=gen_dtype).to(args.device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.set_progress_bar_config(disable=True)
    
    if args.lora_weight_dir_path is not None:
        print('loading LoRA into UNet ....')
        print(f'LoRA path: {args.lora_weight_dir_path}')
        pipe.load_lora_weights(args.lora_weight_dir_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora()
        print('Fused LoRA  ....')

    if args.token_embedding_dir_path is not None and args.token_embedding_dir_path:
        # Load token embeddings from the specified path
        load_token_embedding(pipe.text_encoder, pipe.tokenizer,osp.join(args.token_embedding_dir_path,'token_embedding.pt'))
        print(f"Token embeddings loaded from {args.token_embedding_dir_path}")
    
    
    
    torch.Generator(device=args.device).manual_seed(42)
    
    
# print("unet:")
# print(pipe.unet)
    
    if args.generate_training_data:
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # num_images = 8
        gen_batch_size = 1 # 
        num_images = args.num_gen_images
        print(f'Generating training data... : {num_images} images per concept')
        count = 0
        for single_concept in args.multi_concept:
            generator = None if args.gen_seed is None else torch.Generator(device=args.device).manual_seed(args.gen_seed)
            print(f'set generation seed to {args.gen_seed}')
            for c, t in single_concept:
                count += 1
                already_gen_num_img = 0
                print(f"Generating training data for concept {count}: {c}...")
                c = c.replace('-', ' ')
                output_folder = f"{args.output_dir}/{c}"
                os.makedirs(output_folder, exist_ok=True)
                if t == "object":
                    prompt = f"a photo of the {c}"
                    print(f'Inferencing: {prompt}')
                    # while already_gen_num_img < args.num_gen_images:
                    for i in tqdm(range(0, num_images)):
                        images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.cfg_scale,  generator=generator).images
                        for i, im in enumerate(images):
                            save_path = f"{output_folder}/{prompt.replace(' ', '-')}_{already_gen_num_img}.jpg"
                            im.save(save_path)
                            print(f"Saved image to {save_path}")
                            already_gen_num_img += 1
                            if already_gen_num_img >= args.num_gen_images:
                                break

    
        
                elif t == "style":
                    prompt = f"a photo in the style of {c}"
                    print(f'Inferencing: {prompt}')
                    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
                    for i, im in enumerate(images):
                        im.save(f"{output_folder}/{prompt.replace(' ', '-')}_{i}.jpg")
                else:
                    raise ValueError("unknown concept type.")
                del images
                torch.cuda.empty_cache()
                gc.collect()
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        num_images = args.num_images
        output_folder = f"{args.output_dir}/generated_images"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        prompt = args.prompt
        images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5, num_images_per_prompt=num_images).images
        for i, im in enumerate(images):
            im.save(f"{output_folder}/o_{prompt.replace(' ', '-')}_{i}.jpg")  
        
        torch.cuda.empty_cache()
        gc.collect()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_images', type=int, default=3)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = 30
    model_id = args.model_path
    output_dir = args.save_path
    num_images = args.num_images
    prompt = args.prompt
    
    main(OmegaConf.create({
        "pretrained_model_name_or_path": model_id,
        "generate_training_data": False,
        "device": device,
        "steps": steps,
        "output_dir": output_dir,
        "num_images": num_images,
        "prompt": prompt,
    }))