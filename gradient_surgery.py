
import torch
from typing import Dict, Optional, Callable


import torch
import torch.nn.functional as F
from typing import Dict, Optional
from diffusers.models.attention_processor import AttnProcessor2_0

class AttentionGradientHook(AttnProcessor2_0):
    """
    A gradient-hook-enabled variant of AttnProcessor2_0 for attention modules.

    Exposes per-head attention outputs (`head_outputs`, shape: B×H×Q×Dh) so you can
    read/modify their gradients (e.g., conflict-aware projection, head-wise weighting).
    Forward behavior remains identical to AttnProcessor2_0.

    Works for both self- and cross-attention:
      • If encoder_hidden_states is None → self-attention.
      • Else → cross-attention.

    Access API:
      • iter_hooks() → yields (id(attn), head_outputs)
      • get(attn)    → head_outputs for a specific module (or None)
      • clear()      → clears the cache between steps
    """

    def __init__(self):
        super().__init__()
        self.cache: Dict[int, torch.Tensor] = {}

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
        residual = hidden_states
        input_ndim = hidden_states.ndim

        # optional spatial norm (UNet blocks)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # flatten HW → sequence if needed
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        # self- vs cross-attn
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, seq_len, _ = encoder_hidden_states.shape

        # attention mask → (B, H, Q, K)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # optional group norm pre-attn
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # projections
        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        # (B, H, Q, Dh) etc.
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # per-head attention output (B, H, Q, Dh)
        head_outputs = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # keep autograd path for gradient manipulation downstream
        head_outputs.retain_grad()
        self.cache[id(attn)] = head_outputs

        # merge heads → to_out (same as stock)
        hidden_states = head_outputs.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # reshape back if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, c, h, w)

        # residual + rescale
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    # ---- accessors ----
    def iter_hooks(self):
        """Yield (id(attn), head_outputs) pairs for each attention layer encountered in the last forward."""
        for k, head_outputs in list(self.cache.items()):
            yield k, head_outputs

    def get(self, attn):
        """Get head_outputs for a specific attention module (or None)."""
        return self.cache.get(id(attn), None)

    def clear(self):
        """Clear stored references between steps (recommended each training step)."""
        self.cache.clear()




def _flatten_all(grads: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Flatten and concat all tensors to a single vector."""
    vecs = []
    device = None
    for t in grads.values():
        if t is None:
            continue
        device = t.device
        vecs.append(t.reshape(-1))
    if len(vecs) == 0:
        return torch.tensor([], device=device or "cpu")
    return torch.cat(vecs, dim=0)

def _per_head_flat(g: torch.Tensor) -> torch.Tensor:
    """(B, H, Q, Dh) → (H, B*Q*Dh) for head-wise dot products."""
    return g.permute(1, 0, 2, 3).contiguous().view(g.size(1), -1)

@torch.no_grad()
def generalize_gradient_projection(
    grads_A: Dict[int, torch.Tensor],   # id(attn) → dL_A/d(head_outputs)  (B,H,Q,Dh)
    grads_B: Dict[int, torch.Tensor],   # id(attn) → dL_B/d(head_outputs)  (B,H,Q,Dh)
    *,
    param_group_type: str = "attn_head",   # "global" or "attn_head"
    projection_mode: str = "hard",         # "hard" or "soft"
    rho: Optional[float] = None,           # soft mode: constant ρ in [0,1]
    rho_from_cos: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # soft mode: ρ = f(cos)
) -> Dict[int, torch.Tensor]:
    """
    Combine two gradients via conflict-aware projection.

    param_group_type:
      - "global"    : one projection decision for all modules/heads (coarse)
      - "attn_head" : projection per attention head (fine-grained)

    projection_mode:
      - "hard" : full orthogonal projection when dot<0
      - "soft" : fractional projection (ρ ∈ [0,1]), via constant rho or rho_from_cos

    Returns dict id(attn) → combined gradient w.r.t. head_outputs (same shape as inputs).
    """
    out: Dict[int, torch.Tensor] = {}

    # ===== GLOBAL =====
    if param_group_type == "global":
        gA = _flatten_all(grads_A)
        gB = _flatten_all(grads_B)

        if gA.numel() == 0 or gB.numel() == 0:
            for k, g in grads_A.items():
                out[k] = g
            return out

        dot = torch.dot(gA, gB)

        if dot < 0.0:
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
                raise ValueError(f"Unknown projection_mode: {projection_mode}")

            scale = rho_val * (dot / nB2)
        else:
            scale = 0.0

        for k in grads_A.keys():
            gA_t, gB_t = grads_A[k], grads_B[k]
            if gA_t is None or gB_t is None:
                continue
            g_pc = gA_t - scale * gB_t
            g_pc = g_pc + gB_t
            out[k] = g_pc
        return out

    # ===== ATTENTION-HEAD =====
    elif param_group_type == "attn_head":
        for k in grads_A.keys():
            gA = grads_A[k]
            gB = grads_B[k]
            if gA is None or gB is None:
                continue

            gAh = _per_head_flat(gA)  # (H, *)
            gBh = _per_head_flat(gB)  # (H, *)

            dot = (gAh * gBh).sum(dim=1)              # (H,)
            nB2 = (gBh * gBh).sum(dim=1) + 1e-12      # (H,)

            if projection_mode == "hard":
                rho_h = torch.ones_like(dot)

            elif projection_mode == "soft":
                if rho_from_cos is not None:
                    nA = torch.sqrt((gAh * gAh).sum(dim=1) + 1e-12)
                    nB = torch.sqrt(nB2)
                    cos = dot / (nA * nB + 1e-12)     # (H,)
                    rho_h = rho_from_cos(cos).clamp(0.0, 1.0)
                else:
                    if rho is None:
                        rho_h = torch.ones_like(dot)  # fallback to hard
                    else:
                        rho_h = torch.full_like(dot, float(rho)).clamp(0.0, 1.0)
            else:
                raise ValueError(f"Unknown projection_mode: {projection_mode}")

            conflict = (dot < 0.0)
            scale = torch.zeros_like(dot)
            scale[conflict] = rho_h[conflict] * dot[conflict] / nB2[conflict]

            g_pc = gA - scale.view(1, -1, 1, 1) * gB
            g_pc = g_pc + gB
            out[k] = g_pc

        return out

    else:
        raise ValueError(f"param_group_type must be 'global' or 'attn_head', got: {param_group_type}")
