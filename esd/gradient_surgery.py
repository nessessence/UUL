
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Callable

N_HEADS = 8  # adjust if needed

# --- parameter utils ---

def learnable_param_dict(module):
    return {n: p for n, p in module.named_parameters() if p.requires_grad}

@torch.no_grad()
def collect_param_grads(module, param_names=None):
    out = {}
    params = learnable_param_dict(module)
    iterable = (param_names if param_names is not None else params.keys())
    for n in iterable:
        p = params.get(n)
        if p is None:
            continue
        g = p.grad
        out[n] = (None if g is None else g.detach().clone())
    return out

@torch.no_grad()
def zero_param_grads(module, param_names=None, set_to_none=False):
    params = learnable_param_dict(module)
    iterable = (param_names if param_names is not None else params.keys())
    for n in iterable:
        p = params.get(n)
        if p is None or p.grad is None:
            continue
        if set_to_none:
            p.grad = None
        else:
            p.grad.zero_()

@torch.no_grad()
def param_grad_stats(grads):
    stats = {}
    for n, g in grads.items():
        stats[n] = dict(norm=0.0, numel=0) if g is None else dict(norm=float(g.norm().item()), numel=g.numel())
    return stats

# --- name helpers ---

def _is_qkv_weight(name: str) -> bool:
    return any(x in name for x in (".to_q.weight", ".to_k.weight", ".to_v.weight"))

def _is_o_weight(name: str) -> bool:
    return (".to_out.weight" in name) or (".to_out.0.weight" in name)

def _is_o_bias(name: str) -> bool:
    return (".to_out.bias" in name) or (".to_out.0.bias" in name)

# --- core: head-wise PCGrad on [nH, M] ---

def _pcgrad_headwise(
    A_h: torch.Tensor,  # [nH, M]
    B_h: torch.Tensor,  # [nH, M]
    *,
    projection_mode: str,
    rho_from_cos: Optional[Callable[[torch.Tensor], torch.Tensor]],
    rho_const: Optional[float],
    eps: float = 1e-12,
) -> torch.Tensor:
    dorig = A_h.dtype
    A = A_h.to(torch.float32)
    B = B_h.to(torch.float32)

    dot = (A * B).sum(dim=1)                               # [nH]
    cos = F.cosine_similarity(A, B, dim=1, eps=eps)        # [nH]
    nB2 = (B * B).sum(dim=1).clamp_min(eps)                # [nH]
    
    # print('cosine similarity per head ', cos)

    if projection_mode == "hard":
        rho_h = torch.ones_like(dot)
    elif projection_mode == "soft":
        if rho_from_cos is not None:
            rho_h = rho_from_cos(cos).clamp(0.0, 1.0)
        else:
            rho_h = torch.full_like(dot, 1.0 if rho_const is None else float(rho_const))
    else:
        raise ValueError("projection_mode must be 'hard' or 'soft'")

    conflict = cos < 0.0
    scale_h = torch.zeros_like(dot)
    scale_h[conflict] = rho_h[conflict] * dot[conflict] / nB2[conflict]

    A_proj = (A - scale_h[:, None] * B).to(dorig)          # [nH, M]
    return A_proj

# --- main ---

@torch.no_grad()
def generalize_gradient_projection(
    grads_A: Dict[str, torch.Tensor],
    grads_B: Dict[str, torch.Tensor],
    *,
    param_group_type: str = "attn_head",       # "global" | "layer" | "neuron" | "attn_head"
    projection_mode: Optional[str] = "hard",   # "hard" | "soft" | "none" | None
    rho: Optional[float] = None,
    rho_from_cos: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    scale: float = 1.0, # gradient_projection_preserve_scale
) -> Dict[str, torch.Tensor]:
    """
    Returns resolved grads after conflict handling.
    - 'none'      → gA + scale*gB
    - 'global'    → single projection decision for all params
    - 'layer'     → per-parameter (flatten layer, 1 decision per param)
    - 'neuron'    → per-row vector for 2D weights; per-element for 1D biases
    - 'attn_head' → head-wise by name (q/k/v rows; o columns; skip o.bias)
    """
    out: Dict[str, torch.Tensor] = {}
    print(f"Applying gradient projection: param_group_type={param_group_type}, projection_mode={projection_mode}, scale={scale}")

    # No projection → simple combine
    if projection_mode is None or str(projection_mode).lower() == "none":
        for param_name in grads_A.keys() | grads_B.keys():
            gA, gB = grads_A.get(param_name), grads_B.get(param_name)
            if gA is None and gB is None:
                continue
            out[param_name] = gA if gB is None else (scale * gB if gA is None else gA + scale * gB)
        return out

    eps = 1e-12

    # GLOBAL projection
    if param_group_type == "global":
        def _flatten_all(grads: Dict[str, torch.Tensor]) -> torch.Tensor:
            vecs, device = [], None
            for t in grads.values():
                if t is None:
                    continue
                device = t.device
                vecs.append(t.reshape(-1))
            return torch.cat(vecs, dim=0) if vecs else torch.tensor([], device=device or "cpu")

        gA_all = _flatten_all(grads_A)
        gB_all = _flatten_all(grads_B)

        if gA_all.numel() == 0 or gB_all.numel() == 0:
            for param_name in grads_A.keys() | grads_B.keys():
                gA_t, gB_t = grads_A.get(param_name), grads_B.get(param_name)
                if gA_t is None and gB_t is None:
                    continue
                out[param_name] = gA_t if gB_t is None else (scale * gB_t if gA_t is None else gA_t + scale * gB_t)
            return out

        dot = torch.dot(gA_all, gB_all)
        cos = dot / (torch.norm(gA_all) * torch.norm(gB_all) + eps)
        print(f"grad dot: {float(dot):.6f}, cosine: {float(cos):.6f}")

        if cos < 0.0:
            nB2 = torch.dot(gB_all, gB_all) + eps
            if projection_mode == "hard":
                rho_val = 1.0
            else:
                rho_val = float(rho_from_cos(cos.view(1)).item()) if rho_from_cos else (1.0 if rho is None else float(rho))
            scale_proj = rho_val * (dot / nB2)
        else:
            scale_proj = 0.0

        for param_name, gA_t in grads_A.items():
            gB_t = grads_B.get(param_name)
            if gA_t is None or gB_t is None:
                continue
            g_pc = gA_t - scale_proj * gB_t
            out[param_name] = g_pc + scale * gB_t
        return out

    # LAYER projection: one decision per parameter tensor (flatten)
    if param_group_type == "layer":
        for param_name in grads_A.keys():
            gA = grads_A[param_name]
            gB = grads_B.get(param_name)
            if gA is None or gB is None:
                continue

            A = gA.to(torch.float32).reshape(1, -1)  # [1, M]
            B = gB.to(torch.float32).reshape(1, -1)  # [1, M]

            dot = (A * B).sum(dim=1)                            # [1]
            cos = F.cosine_similarity(A, B, dim=1, eps=eps)     # [1]
            nB2 = (B * B).sum(dim=1).clamp_min(eps)             # [1]
            
            print('Layer ', param_name, ' cosine similarity ', cos)
            

            if projection_mode == "hard":
                rho_val = torch.ones_like(dot)
            else:  # "soft"
                rho_val = rho_from_cos(cos).clamp(0, 1) if rho_from_cos else torch.full_like(dot, 1.0 if rho is None else float(rho))

            conflict = cos < 0
            scale_proj = torch.zeros_like(dot)
            scale_proj[conflict] = rho_val[conflict] * dot[conflict] / nB2[conflict]  # [1]

            Gpc = (A - scale_proj[:, None] * B).reshape(gA.shape).to(gA.dtype)
            out[param_name] = Gpc + scale * gB
        return out

    # NEURON projection:
    #   - for 2D weights [R, C]: per-row vectors length C
    #   - for 1D biases [L]:     per-element (implemented as length-1 vectors)
    if param_group_type == "neuron":
        for param_name in grads_A.keys():
            gA = grads_A[param_name]
            gB = grads_B.get(param_name)
            if gA is None or gB is None:
                continue

            if gA.dim() == 2:  # [R, C] → row-wise projection
                R, C = gA.shape
                A = gA.to(torch.float32)
                B = gB.to(torch.float32)

                dot = (A * B).sum(dim=1)                        # [R]
                cos = F.cosine_similarity(A, B, dim=1, eps=eps) # [R] : 320,640,1280
                nB2 = (B * B).sum(dim=1).clamp_min(eps)         # [R]
                
                # print(cos.shape)
                # print('Neuron ', param_name, ' cosine similarity per row ', cos)

                if projection_mode == "hard":
                    rho_v = torch.ones_like(dot)
                else:
                    rho_v = rho_from_cos(cos).clamp(0, 1) if rho_from_cos else torch.full_like(dot, 1.0 if rho is None else float(rho))

                conflict = cos < 0
                scale_v = torch.zeros_like(dot)
                scale_v[conflict] = rho_v[conflict] * dot[conflict] / nB2[conflict]  # [R]

                Gpc = (A - scale_v[:, None] * B).to(gA.dtype)  # [R, C]
                out[param_name] = Gpc + scale * gB
                continue

            if gA.dim() == 1:  # [L] → element-wise (length-1 vectors)
                L = gA.shape[0]
                A = gA.to(torch.float32).unsqueeze(1)          # [L, 1]
                B = gB.to(torch.float32).unsqueeze(1)          # [L, 1]

                dot = (A * B).sum(dim=1)                        # [L]
                cos = F.cosine_similarity(A, B, dim=1, eps=eps) # [L]
                nB2 = (B * B).sum(dim=1).clamp_min(eps)         # [L]

                if projection_mode == "hard":
                    rho_v = torch.ones_like(dot)
                else:
                    rho_v = rho_from_cos(cos).clamp(0, 1) if rho_from_cos else torch.full_like(dot, 1.0 if rho is None else float(rho))

                conflict = cos < 0
                scale_v = torch.zeros_like(dot)
                scale_v[conflict] = rho_v[conflict] * dot[conflict] / nB2[conflict]  # [L]

                Gpc = (A - scale_v[:, None] * B).squeeze(1).to(gA.dtype)            # [L]
                out[param_name] = Gpc + scale * gB
                continue

            # fallback (e.g., unexpected dims)
            out[param_name] = gA + scale * gB

        return out

    # ATTENTION-HEAD projection (name-driven; your existing behavior)
    if param_group_type == "attn_head":
        for param_name in grads_A.keys():
            gA = grads_A[param_name]
            gB = grads_B.get(param_name)
            if gA is None or gB is None:
                continue

            # skip to_out bias (shared across heads)
            if _is_o_bias(param_name):
                out[param_name] = gA + scale * gB
                continue

            # only handle 2D weights head-wise; fallback otherwise
            if gA.dim() != 2:
                out[param_name] = gA + scale * gB
                continue

            R, C = gA.shape

            if _is_qkv_weight(param_name):
                Dh = R // N_HEADS
                A_h = gA.view(N_HEADS, Dh * C)
                B_h = gB.view(N_HEADS, Dh * C)
                
                # print('A_h shape ', A_h.shape, A_h.norm())

                A_h_pc = _pcgrad_headwise(
                    A_h, B_h,
                    projection_mode=projection_mode,
                    rho_from_cos=rho_from_cos,
                    rho_const=rho,
                    eps=eps,
                )

                # print('A_h_pc shape after PCGrad', A_h_pc.shape, A_h_pc.norm())

                Gpc = A_h_pc.view(N_HEADS, Dh, C).reshape(R, C)
                out[param_name] = Gpc + scale * gB
                continue

            if _is_o_weight(param_name):
                Dh = C // N_HEADS
                # Transpose so heads are rows → symmetric with q/k/v
                A_h = gA.t().contiguous().view(N_HEADS, Dh * R)  # [nH, Dh*R]
                B_h = gB.t().contiguous().view(N_HEADS, Dh * R)  # [nH, Dh*R]

                A_h_pc = _pcgrad_headwise(
                    A_h, B_h,
                    projection_mode=projection_mode,
                    rho_from_cos=rho_from_cos,
                    rho_const=rho,
                    eps=eps,
                )
                # Back to [R, C]
                Gpc = A_h_pc.view(N_HEADS, Dh, R).reshape(C, R).t().contiguous()
                out[param_name] = Gpc + scale * gB
                continue

            # fallback
            out[param_name] = gA + scale * gB

        return out

    raise ValueError("param_group_type must be 'global' | 'layer' | 'neuron' | 'attn_head'")


@torch.no_grad()
def inject_resolved_grads_by_name(
    module: torch.nn.Module,
    resolved_grads: Dict[str, Optional[torch.Tensor]],
    *,
    clone: bool = True,
    clear_missing: bool = False,
) -> None:
    """
    Inject resolved gradients (name → tensor) back into a module.

    Args:
        module:           e.g., your UNet (any nn.Module).
        resolved_grads:   dict[param_name] = Tensor or None.
        clone:            If True, clone() tensors before writing.
        clear_missing:    If True, set .grad=None for params not listed.
    """
    name_to_param = {n: p for n, p in module.named_parameters()}

    # Optionally clear grads for parameters not mentioned in the resolved_grads
    if clear_missing:
        for n, p in name_to_param.items():
            if n not in resolved_grads:
                p.grad = None

    # Write grads
    for name, g_new in resolved_grads.items():
        p = name_to_param.get(name)
        if p is None: raise KeyError(f"[inject_resolved_grads_by_name] Parameter '{name}' not found in module."
)

        if g_new is None:
            p.grad = None
            continue

        g = g_new.detach()
        assert g.shape == p.shape, (f"[inject_resolved_grads_by_name] Shape mismatch for '{name}': "f"resolved {tuple(g.shape)} vs param {tuple(p.shape)}")

        # Match dtype/device
        if g.device != p.device: g = g.to(p.device)
        if g.dtype != p.dtype: g = g.to(p.dtype)

        p.grad = g.clone() if clone else g
        
        
        # “Do not let PyTorch build a computation graph that tries to compute gradients of this gradient.”
        if p.grad.requires_grad:
            p.grad.requires_grad_(False)
            
            
            
            

@torch.no_grad()
def do_grad_injection(
    module: torch.nn.Module,
    resolved_grads: Dict[str, Optional[torch.Tensor]],
    *,
    show_per_param: bool = True,
) -> None:
    """
    Print mean and std of gradients before and after inject_resolved_grads_by_name().
    """
    def _grad_stats(g: Optional[torch.Tensor]):
        if g is None:
            return (None, None)
        g_ = g.detach().float()
        return g_.mean().item(), g_.std().item()

    before_stats = {}
    after_stats = {}

    # collect before injection
    for name, p in module.named_parameters():
        before_stats[name] = _grad_stats(p.grad)

    # inject
    inject_resolved_grads_by_name(module, resolved_grads)

    # collect after injection
    for name, p in module.named_parameters():
        after_stats[name] = _grad_stats(p.grad)

    # show results
    print("🧭 Gradient Injection Summary")
    print("--------------------------------------------------")
    if show_per_param:
        for name in resolved_grads.keys():
            m0, s0 = before_stats.get(name, (None, None))
            m1, s1 = after_stats.get(name, (None, None))
            print(f"{name:60s} | "
                  f"before: mean={m0:.5f} std={s0:.5f}  →  "
                  f"after: mean={m1:.5f} std={s1:.5f}")
    else:
        # aggregated summary
        b_means, b_stds, a_means, a_stds = [], [], [], []
        for name in resolved_grads.keys():
            m0, s0 = before_stats.get(name, (None, None))
            m1, s1 = after_stats.get(name, (None, None))
            if m0 is not None: b_means.append(m0); b_stds.append(s0)
            if m1 is not None: a_means.append(m1); a_stds.append(s1)
        print(f"Before injection: mean(μ)={torch.tensor(b_means).mean():.12f}, std(σ)={torch.tensor(b_stds).mean():.12f}")
        print(f"After injection:  mean(μ)={torch.tensor(a_means).mean():.12f}, std(σ)={torch.tensor(a_stds).mean():.12f}")
    print("--------------------------------------------------\n")