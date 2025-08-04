import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_mean_confidence(
    data: dict,
    steps,
    confidence: float = 90,
    xlabel: str = "Training step",
    ylabel: str = "Score",
    title: str | None = None,
    alpha: float = 0.25,
    figsize: tuple[int, int] = (7, 4.5),
):
    """
    Plot mean ± CI curves for multiple experiments.

    Parameters
    ----------
    data : dict[str, list[list[float]]]
        Mapping {experiment_name: runs}.  Each value must be a list (or
        2‑D array) of shape (n_runs, n_steps).
    steps : 1‑D sequence
        X‑axis labels (must match length of a single run).
    confidence : float
        Confidence level in percent (common choices: 99, 97.5, 95, 90).
    xlabel, ylabel, title : str
        Axis labels and figure title.
    alpha : float
        Transparency of the CI band (0–1).
    figsize : tuple[int, int]
        Figure size in inches.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes.
    """
    # critical z for usual two‑sided CIs
    _z_table = {90: 1.645, 95: 1.960, 97.5: 2.241, 99: 2.576}
    if confidence in _z_table:
        z = _z_table[confidence]
    else:                                        # fallback to scipy if available
        try:
            from scipy.stats import norm
            z = norm.ppf(0.5 + confidence / 100 / 2)
        except ImportError as e:
            raise ValueError(
                f"Unsupported confidence={confidence}. "
                f"Install SciPy or use one of {_z_table.keys()}."
            ) from e

    steps = np.asarray(steps)
    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, runs) in enumerate(data.items()):
        runs = np.asarray(runs, dtype=float)
        if runs.ndim != 2:
            raise ValueError(f"{name}: expected shape (n_runs, n_steps), got {runs.shape}")
        if runs.shape[1] != len(steps):
            raise ValueError(f"{name}: len(steps)={len(steps)} ≠ n_steps={runs.shape[1]}")

        mean = runs.mean(0)
        sem  = runs.std(0, ddof=1) / np.sqrt(runs.shape[0])
        ci   = z * sem

        color = color_cycle[i % len(color_cycle)]
        ax.plot(steps, mean, label=name, color=color)
        ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Metric vs. step (±{confidence}% CI)")
    ax.grid(True)
    ax.legend()
    # plt.tight_layout()
    
    
    # ── NEW: note about the shaded band ──────────────────────────
    ci_note = f"Shaded band = ±{confidence}% CI"
    ax.text(0.02, 0.02, ci_note,
            transform=ax.transAxes,          # axes coords (0–1)
            ha="left", va="bottom",
            fontsize=8, color="gray")
    # ─────────────────────────────────────────────────────────────

    plt.tight_layout()
    
    return fig, ax



concepts = ['chiquita','honer','reese','gout']
concepts = ['chiquita','honer','reese']
learn_settings = ['learn', 'relearn', 'relearn_1e-4']


base_cfg = 7.5
training_steps = list(range(0, 1001, 100))

seeds = [0,1,2,3,4]
seeds = [0,1,2]


for concept in concepts:
    
    exp2scores = defaultdict(list)
    
    ref_img_path = f"data_root/data/real_data/{concept}/{concept}-50"
    
    for learn_setting in learn_settings:
        # scores = []
        for seed in seeds:
            seed_tag = '' if seed == 0 else f'.r{seed}'
            if learn_setting == 'learn':
                gen_img_path =  f"data_root/generated/model/ch.ct.l4.kv_{concept}10-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}"
            elif learn_setting == 'relearn':
                gen_img_path =  f"data_root/generated/model/rlct4.reV.{concept}10.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_ch.c.l16.kv_{concept}50-V.r_pr1.00.neg_lr5e-4.ti5e-4_b1g4.s2000/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}"
            elif learn_setting == 'relearn_1e-4':
                gen_img_path =  f"data_root/generated/model/rlct4.reV.{concept}10.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_ch.c.l16.kv_{concept}50-V.r_pr1.00.neg_lr1e-4.ti5e-4_b1g4.s2000/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}"
            exp2scores[learn_setting] += [load_precision_cache(gen_img_path, ref_img_path, n_max_gen_img=50, training_steps=training_steps)]
        print(f"Scores for {concept} - {learn_setting}: {len(exp2scores[learn_setting])}")
        # print(scores)
        # print(f"Scores for {concept} - {learn_setting}: {len(scores)}")

    plot_mean_confidence(exp2scores,training_steps,title=f"{concept}",ylabel='precision')


concepts = ['chiquita','honer','reese','gout']
concepts = ['chiquita','honer','reese'] #,'gout']
learn_settings = ['learn', 'relearn','relearn_1e-4']


base_cfgs = [7.5, 6.0, 4.5, 3.0]
training_steps = list(range(0, 1001, 100))

seeds = [0,1,2,3,4]
seeds = [0,1,2]

for concept in concepts:
    ref_img_path = f"data_root/data/real_data/{concept}/{concept}-50"
    
    for base_cfg in base_cfgs: 
        gen_img_paths = []; labels = []
        for learn_setting in learn_settings:
            for seed in seeds:
                seed_tag = '' if seed == 0 else f'.r{seed}'
                if learn_setting == 'learn':
                    gen_img_paths +=  [f"data_root/generated/model/ch.ct.l4.kv_{concept}10-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}"]
                    labels += [f"learn_{concept}_{seed}"]
                elif learn_setting == 'relearn':
                    gen_img_paths +=  [f"data_root/generated/model/rlct4.reV.{concept}10.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_ch.c.l16.kv_{concept}50-V.r_pr1.00.neg_lr5e-4.ti5e-4_b1g4.s2000/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}"]
                    labels += [f"relearn_{concept}_{seed}"]
                elif learn_setting == 'relearn_1e-4':
                    gen_img_paths +=  [f"data_root/generated/model/rlct4.reV.{concept}10.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_ch.c.l16.kv_{concept}50-V.r_pr1.00.neg_lr1e-4.ti5e-4_b1g4.s2000/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}"]
                    labels += [f"relearn_{concept}_{seed}"]

        print(labels)
    
        all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, steps, labels, device=device, n_max_gen_img=50, method='cmmd',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True)
        all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, steps, labels, device=device, n_max_gen_img=50, method='pr',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True,clear_notebook_output=False,other_params={"neighborhood": 3})
