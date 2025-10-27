
# gradient_surgery.py
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Callable
from diffusers.models.attention_processor import AttnProcessor,AttnProcessor2_0
from diffusers.utils import deprecate


class AttentionGradientHook(AttnProcessor):
    """
    AttnProcessor-compatible hook that exposes per-head outputs for gradient surgery.

    For each attention module, we cache:
      - cache_by_name[name] = per_head_bhd  (shape: B*H, Q, Dh)   ← this tensor is used downstream
      - meta_by_name[name]  = (B, H, Q, Dh)                       ← to reshape grads when needed

    Why (B*H, Q, Dh)?
      We keep the exact tensor (`per_head_bhd`) that flows through the stock path
      (it's passed into batch_to_head_dim and then to_out). Retaining grad on this
      tensor guarantees `.grad` is populated after backward().
    """

    def __init__(self, proc_name: Optional[str] = None):
        super().__init__()
        self.proc_name = proc_name
        self.cache_by_name: Dict[str, torch.Tensor] = {}  # name -> per_head_bhd (B*H, Q, Dh)
        self.meta_by_name: Dict[str, tuple] = {}          # name -> (B, H, Q, Dh)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # ---- parity with upstream deprecation behavior ----
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecate(
                "scale",
                "1.0.0",
                "The `scale` argument is deprecated and ignored. Pass cross_attention_kwargs to the pipeline instead.",
            )

        residual = hidden_states

        # optional spatial norm (used in some UNet blocks)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # flatten 2D spatial to sequence if needed
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None  # placeholders for restore

        # self- vs cross-attention key/value states
        if encoder_hidden_states is None:
            kv_states = hidden_states
        else:
            kv_states = (
                encoder_hidden_states
                if not attn.norm_cross
                else attn.norm_encoder_hidden_states(encoder_hidden_states)
            )

        # attention mask prepared the stock way
        sequence_length = (hidden_states.shape if encoder_hidden_states is None else kv_states.shape)[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # optional group norm pre-attn
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        
        print(f"attn heads: {attn.heads}")
        # print(f"attn head_dim: {attn.head_dim}")
        print(f"attn.to_q weight shape: {attn.to_q.weight.shape}")
        
        print(f"attn.to_k weight shape: {attn.to_k.weight.shape}")
        print(f"attn.to_v weight shape: {attn.to_v.weight.shape}")
        
        print("Hidden states shape before projections:", hidden_states.shape)
        print("kv_states shape before projections:", kv_states.shape)

        # linear projections
        query = attn.to_q(hidden_states)
        key   = attn.to_k(kv_states)
        value = attn.to_v(kv_states)

        print(f"Query shape: {query.shape}")
        print(f"Key shape: {key.shape}")
        print(f"Value shape: {value.shape}")

        # reshape to (B*H, Q, Dh) using stock helpers (keeps parity with diffusers)
        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # stock attention scores (no flash/mem-efficient kernels here)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # (B*H, Q, K)
        
        print(f"Attention probs shape: {attention_probs.shape}")

        # per-head outputs BEFORE merging heads
        # shape: (B*H, Q, Dh)
        per_head_bhd = torch.bmm(attention_probs, value)
        
        
        print(f'per_head_bhd shape: {per_head_bhd.shape}')

        # record shape meta for later reshape to (B, H, Q, Dh)
        B = batch_size
        H = attn.heads # 8
        Q = per_head_bhd.shape[1] # 4096,1024,256
        Dh = per_head_bhd.shape[2] # 40,80,160
        
        # print(self.proc_name,Dh)

        # retain grad on the tensor that actually flows to the output path
        per_head_bhd.retain_grad()

        # cache for this processor
        if self.proc_name is not None:
            self.cache_by_name[self.proc_name] = per_head_bhd
            self.meta_by_name[self.proc_name] = (B, H, Q, Dh)

        # merge heads -> (B, Q, H*Dh), then the output projection + dropout
        hidden_states = attn.batch_to_head_dim(per_head_bhd)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # restore 2D spatial if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, channel, height, width)

        # residual connection and rescale (stock)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor


        print(f'final hidden state output: {hidden_states.shape}')

        return hidden_states

    def clear(self):
        self.cache_by_name.clear()
        self.meta_by_name.clear()

# class AttentionGradientHook(AttnProcessor):
#     """
#     AttnProcessor-compatible hook that exposes per-head outputs (B,H,Q,Dh) for gradient surgery.
#     Each instance is bound to ONE attention module and tagged with its processor name (proc_name).
#     After forward:
#         proc.cache_by_name[proc.proc_name] -> tensor (B,H,Q,Dh) with .grad populated after backward()
#     """
#     def __init__(self, proc_name: Optional[str] = None):
#         super().__init__()
#         self.proc_name = proc_name
#         self.cache_by_name: Dict[str, torch.Tensor] = {}

#     def __call__(
#         self,
#         attn,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         temb: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         # --- stock AttnProcessor path start ---
#         if len(args) > 0 or kwargs.get("scale", None) is not None:
#             # keep parity with upstream deprecation behavior
#             from diffusers.utils import deprecate
#             deprecate(
#                 "scale", "1.0.0",
#                 "The `scale` argument is deprecated and ignored. Pass cross_attention_kwargs to the pipeline instead."
#             )

#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim
#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#         else:
#             batch_size, channel, height, width = hidden_states.shape[0], None, None, None  # placeholders

#         # self vs cross
#         if encoder_hidden_states is None:
#             kv_states = hidden_states
#         else:
#             kv_states = encoder_hidden_states if not attn.norm_cross else attn.norm_encoder_hidden_states(encoder_hidden_states)

#         # attention mask (stock helper)
#         sequence_length = (hidden_states.shape if encoder_hidden_states is None else kv_states.shape)[1]
#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#         # optional pre-attn norm
#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         # projections
#         query = attn.to_q(hidden_states)
#         key   = attn.to_k(kv_states)
#         value = attn.to_v(kv_states)

#         # (B*H, Q, Dh) with stock helpers
#         query = attn.head_to_batch_dim(query)
#         key   = attn.head_to_batch_dim(key)
#         value = attn.head_to_batch_dim(value)

#         # attention scores (stock API; no flash/mem-efficient kernels here)
#         attention_probs = attn.get_attention_scores(query, key, attention_mask)

#         # --- per-head outputs BEFORE merging heads ---
#         # shape: (B*H, Q, Dh)
#         per_head_bhd = torch.bmm(attention_probs, value)

#         # reshape to (B, H, Q, Dh) for gradient surgery & retain grads
#         H = attn.heads
#         Q, Dh = per_head_bhd.shape[1], per_head_bhd.shape[2]
#         head_outputs = per_head_bhd.view(batch_size, H, Q, Dh)
#         per_head_bhd.retain_grad()
#         head_outputs.retain_grad(); 
#         if self.proc_name is not None:
#             self.cache_by_name[self.proc_name] = head_outputs
#             # self.cache_by_name[self.proc_name] = per_head_bhd

#         # merge heads → (B, Q, H*Dh), then out proj + dropout (stock)
#         hidden_states = attn.batch_to_head_dim(per_head_bhd)  # (B, Q, H*Dh)
#         hidden_states = attn.to_out[0](hidden_states)
#         hidden_states = attn.to_out[1](hidden_states)

#         # reshape back if we started with 4D
#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor
#         # --- stock AttnProcessor path end ---
#         return hidden_states

#     def clear(self):
#         self.cache_by_name.clear()
        
        
# class AttentionGradientHook(AttnProcessor):
#     """
#     AttnProcessor-compatible hook that exposes per-head outputs (B,H,Q,Dh) for gradient surgery.
#     Each instance is bound to ONE attention module and tagged with its processor name (proc_name).
#     After forward:
#         proc.cache_by_name[proc.proc_name] -> tensor (B,H,Q,Dh) with .grad populated after backward()
#     """
#     def __init__(self, proc_name: Optional[str] = None):
#         super().__init__()
#         self.proc_name = proc_name
#         self.cache_by_name: Dict[str, torch.Tensor] = {}

#     def __call__(
#         self,
#         attn,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         temb: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         # --- stock AttnProcessor path start ---
#         if len(args) > 0 or kwargs.get("scale", None) is not None:
#             # keep parity with upstream deprecation behavior
#             from diffusers.utils import deprecate
#             deprecate(
#                 "scale", "1.0.0",
#                 "The `scale` argument is deprecated and ignored. Pass cross_attention_kwargs to the pipeline instead."
#             )

#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim
#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#         else:
#             batch_size, channel, height, width = hidden_states.shape[0], None, None, None  # placeholders

#         # self vs cross
#         if encoder_hidden_states is None:
#             kv_states = hidden_states
#         else:
#             kv_states = encoder_hidden_states if not attn.norm_cross else attn.norm_encoder_hidden_states(encoder_hidden_states)

#         # attention mask (stock helper)
#         sequence_length = (hidden_states.shape if encoder_hidden_states is None else kv_states.shape)[1]
#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#         # optional pre-attn norm
#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         # projections
#         query = attn.to_q(hidden_states)
#         key   = attn.to_k(kv_states)
#         value = attn.to_v(kv_states)

#         # (B*H, Q, Dh) with stock helpers
#         query = attn.head_to_batch_dim(query)
#         key   = attn.head_to_batch_dim(key)
#         value = attn.head_to_batch_dim(value)

#         # attention scores (stock API; no flash/mem-efficient kernels here)
#         attention_probs = attn.get_attention_scores(query, key, attention_mask)

#         # --- per-head outputs BEFORE merging heads ---
#         # shape: (B*H, Q, Dh)
#         per_head_bhd = torch.bmm(attention_probs, value)

#         # reshape to (B, H, Q, Dh) for gradient surgery & retain grads
#         H = attn.heads
#         Q, Dh = per_head_bhd.shape[1], per_head_bhd.shape[2]
#         head_outputs = per_head_bhd.view(batch_size, H, Q, Dh)
#         # head_outputs.retain_grad()
#         per_head_bhd.retain_grad()
#         if self.proc_name is not None:
#             self.cache_by_name[self.proc_name] = per_head_bhd
            
#             # self.cache_by_name[self.proc_name] = head_outputs

#         # merge heads → (B, Q, H*Dh), then out proj + dropout (stock)
#         hidden_states = attn.batch_to_head_dim(per_head_bhd)  # (B, Q, H*Dh)
#         hidden_states = attn.to_out[0](hidden_states)
#         hidden_states = attn.to_out[1](hidden_states)

#         # reshape back if we started with 4D
#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor
#         # --- stock AttnProcessor path end ---
#         return hidden_states

#     def clear(self):
#         self.cache_by_name.clear()

        
        
# # =========  Per-processor hook (each instance knows its own processor name)  =========
# class AttentionGradientHook(AttnProcessor2_0):
#     """
#     Tiny attention processor exposing per-head outputs (B,H,Q,Dh) for gradient surgery.
#     Each instance is created for ONE attention module and tagged with its processor name
#     (e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor").

#     After a forward pass:
#         proc.cache_by_name[proc.proc_name] -> head_outputs tensor (B,H,Q,Dh) with .grad after backward()
#     """
#     def __init__(self, proc_name: Optional[str] = None):
#         super().__init__()
#         self.proc_name = proc_name
#         self.cache_by_name: Dict[str, torch.Tensor] = {}

#     def __call__(
#         self,
#         attn,
#         hidden_states: torch.Tensor,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         temb: Optional[torch.Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> torch.Tensor:
#         residual = hidden_states
#         input_ndim = hidden_states.ndim

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         if input_ndim == 4:
#             b, c, h, w = hidden_states.shape
#             hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         batch_size, seq_len, _ = encoder_hidden_states.shape

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)
#         key   = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim  = inner_dim // attn.heads

#         # (B,H,Q,Dh)
#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         head_outputs = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )
#         # expose for gradient surgery
#         head_outputs.retain_grad()
#         if self.proc_name is not None:
#             self.cache_by_name[self.proc_name] = head_outputs
#             # print(self.proc_name) # up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor
#             # print(self.cache_by_name[self.proc_name])

#         # merge heads → out proj (stock)
#         hidden_states = head_outputs.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)
#         hidden_states = attn.to_out[0](hidden_states)
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, c, h, w)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual
#         hidden_states = hidden_states / attn.rescale_output_factor
#         return hidden_states

#     def clear(self):
#         self.cache_by_name.clear()


# =========  Install one hook PER attention processor (tagged by name)  =========
def install_hook_on_learnables(unet, hook_cls, learnable_param_names=None, **hook_kwargs):
    """
    Install one hook instance per targeted attention processor, each tagged with its own name.
    Returns: sorted list of processor names hooked.
    """
    if learnable_param_names is None:
        target_keys = set(unet.attn_processors.keys())
    else:
        # map param names like "...attn2.to_q.weight" -> "...attn2.processor"
        target_keys = {
            f"{name[:name.index(tag) + len(tag)]}processor"
            for tag in (".attn1.", ".attn2.")
            for name in learnable_param_names
            if tag in name
        }

    new_map = {}
    hooked = []
    for name, proc in unet.attn_processors.items():
        if name in target_keys:
            new_map[name] = hook_cls(proc_name=name, **hook_kwargs)
            hooked.append(name)
        else:
            new_map[name] = proc
    unet.set_attn_processor(new_map)
    return sorted(hooked)


# =========  Helpers: collect / zero / inject / clear by NAME  =========


@torch.no_grad()
@torch.no_grad()
def collect_grads_by_name_from_unet(
    unet,
    hook_cls,
    batch_slice: slice = None,  # e.g., slice(0, B//2) or slice(B//2, B)
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Collect grads as dict name -> (B_slice, H, Q, Dh) or None.
    If batch_slice is None, returns full (B, H, Q, Dh).
    """
    grads = {}
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, hook_cls):
            t = proc.cache_by_name.get(name, None)  # (B*H, Q, Dh)
            if t is None or t.grad is None:
                grads[name] = None
                continue
            B, H, Q, Dh = proc.meta_by_name[name]
            g_full = t.grad.view(B, H, Q, Dh)
            if batch_slice is None:
                grads[name] = g_full.detach().clone()
            else:
                grads[name] = g_full[batch_slice].detach().clone()  # (B_slice, H, Q, Dh)
                # print( g_full.shape,  grads[name].shape) # torch.Size([1, 8, 64, 160]) torch.Size([2, 8, 64, 160]) 
                # print(grads[name].sum())
    return grads


# def collect_grads_by_name_from_unet(unet, hook_cls) -> Dict[str, Optional[torch.Tensor]]:
#     grads = {}
#     for name, proc in unet.attn_processors.items():
#         if isinstance(proc, hook_cls):
#             t = proc.cache_by_name.get(name, None)  # (B*H, Q, Dh)
#             if t is None or t.grad is None:
#                 grads[name] = None
#             else:
#                 B, H, Q, Dh = proc.meta_by_name[name]
#                 grads[name] = t.grad.view(B, H, Q, Dh).detach().clone()
#     return grads

@torch.no_grad()
def zero_cached_grads_in_unet(unet, hook_cls):
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, hook_cls):
            t = proc.cache_by_name.get(name, None)
            if t is not None and t.grad is not None:
                t.grad.zero_()

@torch.no_grad()
def inject_resolved_grads_by_name(unet, hook_cls, resolved_grads: Dict[str, torch.Tensor]):
    tensors, grads = [], []
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, hook_cls):
            t = proc.cache_by_name.get(name, None)  # (B*H, Q, Dh)
            g = resolved_grads.get(name, None)      # (B, H, Q, Dh)
            if t is not None and g is not None:
                B, H, Q, Dh = proc.meta_by_name[name]
                grads.append(g.view(B*H, Q, Dh))
                tensors.append(t)
    if tensors:
        torch.autograd.backward(tensors, grad_tensors=grads)




# @torch.no_grad()
# def collect_grads_by_name_from_unet(unet, hook_cls) -> Dict[str, Optional[torch.Tensor]]:
#     grads = {}
#     for name, proc in unet.attn_processors.items():
#         if isinstance(proc, hook_cls):
#             t = proc.cache_by_name.get(name, None)
#             grads[name] = (t.grad.detach().clone() if (t is not None and t.grad is not None) else None)
#     return grads

# @torch.no_grad()
# def zero_cached_grads_in_unet(unet, hook_cls):
#     for name, proc in unet.attn_processors.items():
#         if isinstance(proc, hook_cls):
#             t = proc.cache_by_name.get(name, None)
#             if t is not None and t.grad is not None:
#                 t.grad.zero_()

# @torch.no_grad()
# def inject_resolved_grads_by_name(unet, hook_cls, resolved_grads: Dict[str, torch.Tensor]):
#     tensors, grads = [], []
#     for name, proc in unet.attn_processors.items():
#         if isinstance(proc, hook_cls):
#             t = proc.cache_by_name.get(name, None)
#             g = resolved_grads.get(name, None)
#             if t is not None and g is not None:
#                 tensors.append(t)
#                 grads.append(g)
#     if tensors:
#         torch.autograd.backward(tensors, grad_tensors=grads)

@torch.no_grad()
def clear_hook_caches(unet, hook_cls):
    for _, proc in unet.attn_processors.items():
        if isinstance(proc, hook_cls):
            proc.clear()


# =========  Gradient projection (PCGrad-style) with a single `scale` for B  =========
def _flatten_all(grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    vecs, device = [], None
    for t in grads.values():
        if t is None:
            continue
        device = t.device
        vecs.append(t.reshape(-1))
    return torch.cat(vecs, dim=0) if vecs else torch.tensor([], device=device or "cpu")

def _per_head_flat(g: torch.Tensor) -> torch.Tensor:
    # (B,H,Q,Dh) -> (H, B*Q*Dh)
    return g.permute(1, 0, 2, 3).contiguous().view(g.size(1), -1)

@torch.no_grad()
def generalize_gradient_projection(
    grads_A: Dict[str, torch.Tensor],   # name -> dL_A/d(head_outputs)
    grads_B: Dict[str, torch.Tensor],   # name -> dL_B/d(head_outputs)
    *,
    param_group_type: str = "attn_head",       # "global" | "attn_head"
    projection_mode: Optional[str] = "hard",   # "hard" | "soft" | "none" | None
    rho: Optional[float] = None,
    rho_from_cos: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    scale: float = 1.0,                        # weight for B (preserve) branch
) -> Dict[str, torch.Tensor]:
    """
    Return `resolved_grads`: a dict of gradients after conflict resolution.
    Only conflicting components are projected; aligned ones are left intact.
    - projection_mode "none"/None -> naive weighted sum (equiv. to (lossA + scale*lossB).backward()).
    - "hard" -> full orthogonal projection when dot < 0.
    - "soft" -> partial projection (rho in [0,1]).
    """
    out: Dict[str, torch.Tensor] = {}
    print(f"Applying gradient projection: param_group_type={param_group_type}, projection_mode={projection_mode}, scale={scale}")
    # No projection → naive weighted sum
    if projection_mode is None or str(projection_mode).lower() == "none":
        for k in grads_A.keys() | grads_B.keys():
            gA, gB = grads_A.get(k), grads_B.get(k)
            if gA is None and gB is None:
                continue
            if gA is None:
                out[k] = scale * gB
            elif gB is None:
                out[k] = gA
            else:
                out[k] = gA + scale * gB
        return out

    # GLOBAL: one decision for all tensors
    if param_group_type == "global":
        gA = _flatten_all(grads_A)
        gB = _flatten_all(grads_B)
        # print(f"Global grad sizes: gA={gA.numel()}, gB={gB.numel()}") # gA=0, gB=11550720
        if gA.numel() == 0 or gB.numel() == 0:
            return grads_A
        dot = torch.dot(gA, gB)  
        # print(gA.shape, gB.shape) # torch.Size([0]) torch.Size([11550720])
        cosine_sim = torch.cosine_similarity(gA.view(1, -1), gB.view(1, -1))
        # print(gA.sum(), gB.sum()) # 11550720, # tensor(-0.0020, device='cuda:3', dtype=torch.float16) tensor(0., device='cuda:3', dtype=torch.float16)   
        print(f"grad dot: {dot.item()}, cosine: {cosine_sim.item()}") # ... 0 ?? one of them is all zero?
        if dot < 0.0:
            print(f"Conflict detected (dot={dot.item():.6f}) in GLOBAL projection.")
            nB2 = torch.dot(gB, gB) + 1e-12
            if projection_mode == "hard":
                rho_val = 1.0
            elif projection_mode == "soft":
                if rho_from_cos is not None:
                    cos = dot / (torch.sqrt(torch.dot(gA, gA) + 1e-12) * torch.sqrt(nB2))
                    rho_val = float(rho_from_cos(cos.view(1)).item())
                else:
                    rho_val = 1.0 if rho is None else float(rho)
            else:
                raise ValueError("Unknown projection_mode")
            scale_proj = rho_val * (dot / nB2)
        else:
            scale_proj = 0.0

        for k, gA_t in grads_A.items():
            gB_t = grads_B.get(k, None)
            if gA_t is None or gB_t is None:
                continue
            g_pc = gA_t - scale_proj * gB_t   # project A wrt B if conflicting
            out[k] = g_pc + scale * gB_t      # add weighted B
        return out

    # ATTENTION-HEAD: per-head decision
    elif param_group_type == "attn_head":
        for k in grads_A.keys():
            gA = grads_A[k]
            gB = grads_B.get(k, None)
            if gA is None or gB is None:
                continue
            # print(gA.shape, gB.shape) # [1, 8, 4096, 40]
            gAh = _per_head_flat(gA)
            gBh = _per_head_flat(gB)
            dot = (gAh * gBh).sum(dim=1)
            nB2 = (gBh * gBh).sum(dim=1) + 1e-12
            
            cosine_sim = F.cosine_similarity(gAh, gBh, dim=1)
            # print(dot.shape) # torch.Size([8]) 
            print(cosine_sim)
            # print(gAh.shape, gBh.shape) # torch.Size([8, 204800]) torch.Size([8, 204800])

            if projection_mode == "hard":
                rho_h = torch.ones_like(dot)
            elif projection_mode == "soft":
                if rho_from_cos is not None:
                    nA = torch.sqrt((gAh * gAh).sum(dim=1) + 1e-12)
                    nB = torch.sqrt(nB2)
                    cos = dot / (nA * nB + 1e-12)
                    rho_h = rho_from_cos(cos).clamp(0.0, 1.0)
                else:
                    rho_h = torch.full_like(dot, 1.0 if rho is None else float(rho))
            else:
                raise ValueError("Unknown projection_mode")

            conflict = (dot < 0.0)
            scale_proj = torch.zeros_like(dot)
            scale_proj[conflict] = rho_h[conflict] * dot[conflict] / nB2[conflict]

            g_pc = gA - scale_proj.view(1, -1, 1, 1) * gB  # project only conflicting heads
            out[k] = g_pc + scale * gB
        return out

    else:
        raise ValueError("param_group_type must be 'global' or 'attn_head'")



 