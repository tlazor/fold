from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Prepare data
layers = np.array([1, 3, 6, 9, 12])

overlap = {
    "mut_int":  {"p_coef": [-0.542, -0.521, -0.456, -0.447, -0.499],
                 "s_coef": [-0.552, -0.559, -0.520, -0.526, -0.550],
                 "p_pval": [0.000, 0.000, 0.001, 0.001, 0.000],
                 "s_pval": [0.000, 0.000, 0.000, 0.000, 0.000]},
    "lex_sim":  {"p_coef": [0.547, 0.569, 0.545, 0.542, 0.610],
                 "s_coef": [0.564, 0.576, 0.572, 0.562, 0.618],
                 "p_pval": [0.001, 0.000, 0.001, 0.001, 0.000],
                 "s_pval": [0.000, 0.000, 0.000, 0.000, 0.000]},
    "fsi":     {"p_coef": [0.193, 0.294, 0.340, 0.340, 0.333],
                 "s_coef": [0.387, 0.490, 0.450, 0.462, 0.551],
                 "p_pval": [0.282, 0.097, 0.053, 0.053, 0.058],
                 "s_pval": [0.026, 0.004, 0.009, 0.007, 0.001]},
    "pho_sim":  {"p_coef": [0.542, 0.585, 0.603, 0.580, 0.584],
                 "s_coef": [0.415, 0.506, 0.541, 0.544, 0.562],
                 "p_pval": [0.000, 0.000, 0.000, 0.000, 0.000],
                 "s_pval": [0.000, 0.000, 0.000, 0.000, 0.000]},
}

kl_div = {
    "mut_int":  {"p_coef": [-0.523, -0.505, -0.420, -0.400, -0.389],
                 "s_coef": [-0.545, -0.559, -0.504, -0.506, -0.403],
                 "p_pval": [0.000, 0.000, 0.002, 0.003, 0.004],
                 "s_pval": [0.000, 0.000, 0.000, 0.000, 0.003]},
    "lex_sim":  {"p_coef": [ 0.570,  0.566,  0.514,  0.482,  0.484],
                 "s_coef": [ 0.652,  0.595,  0.528,  0.508,  0.568],
                 "p_pval": [0.000, 0.000, 0.002, 0.003, 0.003],
                 "s_pval": [0.000, 0.000, 0.001, 0.002, 0.000]},
    "fsi":     {"p_coef": [ 0.369,  0.382,  0.408,  0.436,  0.468],
                 "s_coef": [ 0.564,  0.536,  0.492,  0.534,  0.518],
                 "p_pval": [0.035, 0.028, 0.018, 0.011, 0.006],
                 "s_pval": [0.001, 0.001, 0.004, 0.001, 0.002]},
    "pho_sim":  {"p_coef": [ 0.603,  0.624,  0.627,  0.627,  0.618],
                 "s_coef": [ 0.595,  0.578,  0.601,  0.631,  0.547],
                 "p_pval": [0.000, 0.000, 0.000, 0.000, 0.000],
                 "s_pval": [0.000, 0.000, 0.000, 0.000, 0.000]},
}

def plot_2x2(data, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    ((ax_ppval, ax_pcoef), (ax_spval, ax_scoef)) = axes

    for metric, vals in data.items():
        pp = vals['p_pval']
        sp = vals['s_pval']

        # Pearson
        ax_ppval.plot(layers, pp, marker='o', label=metric)
        ax_pcoef.plot(layers, vals['p_coef'], marker='o', label=metric)

        # Spearman
        ax_spval.plot(layers, sp, marker='o', label=metric)
        ax_scoef.plot(layers, vals['s_coef'], marker='o', label=metric)

    # Axis styling
    ax_ppval.set_ylim(0,1)
    ax_ppval.axhline(0.05, color='red', linestyle='--', linewidth=1)
    ax_spval.set_ylim(0,1)
    ax_spval.axhline(0.05, color='red', linestyle='--', linewidth=1)

    ax_pcoef.set_ylim(-1,1)
    ax_scoef.set_ylim(-1,1)

    ax_ppval.set_ylabel('p-value')
    ax_spval.set_ylabel('p-value')
    ax_pcoef.set_ylabel('Coefficient')
    ax_scoef.set_ylabel('Coefficient')

    for ax in (ax_spval, ax_scoef):
        ax.set_xlabel('Layer')

    ax_ppval.set_title('Pearson p-values')
    ax_pcoef.set_title('Pearson coefficients')
    ax_spval.set_title('Spearman p-values')
    ax_scoef.set_title('Spearman coefficients')

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = ax_pcoef.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels))
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    plt.savefig(Path(f"notes/thesis/layer/{title}_pval.png"))
    plt.show()
    plt.clf()

plot_2x2(overlap, 'Overlap correlations across layers')
plot_2x2(kl_div, 'KL divergence correlations across layers')