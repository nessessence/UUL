import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
# from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, retrieve_timesteps

# compat between diffusers versions
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps




if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def parse_generation_phase_parameter(
    phase_string: str,
    orginal_pretrained_weight,
    unlearned_weight,
):
    """
    Parse a compact string spec into a generation_phase_parameter list.

    Example:
        "$PT.500.a photo of person-UL.0.a photo Margot Robbie"
        ->
        [
            {"timestep": 500, "unet_weights": orginal_pretrained_weight, "prompt": "a photo of person"},
            {"timestep": 0, "unet_weights": unlearned_weight, "prompt": "a photo Margot Robbie"},
        ]
    """
    if phase_string is None:
        raise ValueError("phase_string must be a non-empty string.")

    token_to_weight = {"PT": orginal_pretrained_weight, "UL": unlearned_weight}

    cleaned = phase_string.split("*Ph.")[-1]
    if not cleaned:
        raise ValueError("phase_string must be a non-empty string.")

    segments = [seg for seg in cleaned.split("-") if seg]
    if not segments:
        raise ValueError("phase_string did not contain any segments to parse.")

    phases = []
    simplified_phases = []
    for seg in segments:
        parts = seg.split(".", 2)
        if len(parts) != 3:
            raise ValueError(f"Segment '{seg}' is malformed. Expected format TOKEN.TIMESTEP.PROMPT.")

        token, timestep_str, prompt = parts
        if token not in token_to_weight:
            raise ValueError(f"Unknown weight token '{token}'. Expected one of {list(token_to_weight.keys())}.")

        try:
            timestep = int(timestep_str)
        except Exception as exc:
            raise ValueError(f"Could not parse timestep '{timestep_str}' as int.") from exc

        phases.append(
            {"timestep": timestep, "unet_weights": token_to_weight[token], "prompt": prompt.strip()}
        )

        simplified_phases.append(
            {
                "timestep": timestep,
                "unet_weights": "orginal_pretrained_weight" if token == "PT" else "unlearned_weight",
                "prompt": prompt.strip(),
            }
        )

    return phases, simplified_phases


@torch.no_grad()
def custom_call(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    run_from_timestep=0,
    run_till_timestep=None,
    start_latents=None,
    save_every_step_latents: bool = False,
    generation_phase_parameter: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.
    """
    
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
    #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    # if not height or not width:
    #     height = (
    #         self.unet.config.sample_size
    #         if self._is_unet_config_sample_size_int
    #         else self.unet.config.sample_size[0]
    #     )
    #     width = (
    #         self.unet.config.sample_size
    #         if self._is_unet_config_sample_size_int
    #         else self.unet.config.sample_size[1]
    #     )
    #     height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
    
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    # self.check_inputs(
    #     prompt,
    #     height,
    #     width,
    #     callback_steps,
    #     negative_prompt,
    #     prompt_embeds,
    #     negative_prompt_embeds,
    #     ip_adapter_image,
    #     ip_adapter_image_embeds,
    #     callback_on_step_end_tensor_inputs,
    # )

    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

        
    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    phase_settings = None
    phase_prompt_embeds = None
    negative_prompt_embeds = negative_prompt_embeds

    if generation_phase_parameter:
        if prompt_embeds is not None:
            raise ValueError("generation_phase_parameter does not support passing precomputed prompt_embeds.")

        if not isinstance(generation_phase_parameter, (list, tuple)):
            raise ValueError("generation_phase_parameter must be a list of dictionaries describing each phase.")

        phase_settings = []
        for idx, phase in enumerate(generation_phase_parameter):
            if not isinstance(phase, dict):
                raise ValueError("Each entry in generation_phase_parameter must be a dictionary.")
            if "timestep" not in phase:
                raise ValueError("Each generation_phase_parameter entry must include a 'timestep' key.")
            phase_settings.append(
                {
                    "timestep": phase["timestep"],
                    "unet_weights": phase.get("unet_weights"),
                    "prompt": phase.get("prompt", prompt),
                }
            )

        # Sort descending since diffusion timesteps typically count down (e.g., 999 -> 0).
        phase_settings = sorted(phase_settings, key=lambda p: p["timestep"], reverse=True)
        phase_prompt_embeds = []
        for phase in phase_settings:
            phase_prompt = phase["prompt"]
            if phase_prompt is None:
                raise ValueError(
                    "A prompt must be provided either directly or inside each generation_phase_parameter entry."
                )
                
            print(phase_prompt)
            current_prompt_embeds, current_negative_prompt_embeds = self.encode_prompt(
                [phase_prompt]*len(prompt),
                # phase_prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            phase_prompt_embeds.append(current_prompt_embeds)
            if negative_prompt_embeds is None:
                negative_prompt_embeds = current_negative_prompt_embeds
    else:
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

    def _combine_prompt_embeds(pos_prompt_embeds: torch.Tensor) -> torch.Tensor:
        if self.do_classifier_free_guidance:
            return torch.cat([negative_prompt_embeds, pos_prompt_embeds])
        return pos_prompt_embeds

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas
    )
    timesteps = timesteps[run_from_timestep: run_till_timestep]

    # Establish the initial phase (if any) based on the first timestep and load the corresponding assets.
    phase_index = None
    if phase_settings is not None and len(phase_settings) > 0:
        def _get_phase_index(timestep_value):
            ts_val = timestep_value.item() if hasattr(timestep_value, "item") else timestep_value
            for idx, phase in enumerate(phase_settings):
                if ts_val >= phase["timestep"]:
                    return idx
            return len(phase_settings) - 1

        def _apply_phase(idx: int):
            nonlocal prompt_embeds, phase_index
            phase_index = idx
            phase = phase_settings[idx]
            if phase.get("unet_weights") is not None:
                self.unet.load_state_dict(phase["unet_weights"], strict=False)
            phase_prompt = phase_prompt_embeds[idx]
            prompt_embeds = _combine_prompt_embeds(phase_prompt)

        initial_idx = _get_phase_index(timesteps[0])
        _apply_phase(initial_idx)
    else:
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = _combine_prompt_embeds(prompt_embeds)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    if start_latents is not None:
        latents = start_latents
    saved_latents = [] if save_every_step_latents else None
    saved_timesteps = [] if save_every_step_latents else None
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # if self.interrupt:
            #     continue
            
            
            print(f"Step {i+1}/{len(timesteps)}; Timestep: {t}")
            if phase_settings is not None:
                current_phase_idx = _get_phase_index(t)
                print(f" Current Phase Index: {current_phase_idx}, Applied Phase Index: {phase_index}")
                if current_phase_idx != phase_index:
                    _apply_phase(current_phase_idx)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # print('hey')
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if save_every_step_latents:
                saved_latents.append(latents.detach().clone())
                saved_timesteps.append(t.detach().clone() if hasattr(t, "detach") else t)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

    if not output_type == "latent":
        # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
        #     0
        # ]
        
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[
            0
        ]
                
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    
    
    # print(f" mse( image, latents): { ((image - latents)**2).mean()} ") # 0.0 
    # print(f" max abs diff( image, latents): { (image - latents).abs().max()}") # 0.0 
    # print(f"image shape: {image.shape}") # [4, 4, 64, 64]
    # [4, 4, 128, 128] if they are using 1024
    

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        if save_every_step_latents:
            return (image, has_nsfw_concept, saved_latents, saved_timesteps)
        return (image, has_nsfw_concept)

    output = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    output.latents = latents
    output.step_latents = saved_latents
    output.step_timesteps = saved_timesteps
    output.timesteps = timesteps
    return output
