import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln

sns.set_theme(style="whitegrid")

def log_comb(n, k):
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def compute_pass_k_adv_raw(N, c_neg, k):
    # P(fail) = C(c_neg, k) / C(N, k)
    log_p_fail = log_comb(c_neg, k) - log_comb(N, k)
    p_fail = np.exp(log_p_fail) if log_p_fail > -np.inf else 0.0
    
    r_mean = 1.0 - p_fail
    # Bernoulli std
    sigma_b = np.sqrt(r_mean * (1.0 - r_mean) + 1e-8)
    
    a_pos = p_fail / sigma_b
    
    log_p_cond_fail = log_comb(c_neg - 1, k - 1) - log_comb(N - 1, k - 1)
    p_cond_fail = np.exp(log_p_cond_fail) if log_p_cond_fail > -np.inf else 0.0
    a_neg = (p_fail - p_cond_fail) / sigma_b
    
    return a_pos, a_neg

def plot_specific_sample_distribution(N=32, k=4, c_maj=20):
    c_neg = N - c_maj
    a_pos_base, a_neg_base = compute_pass_k_adv_raw(N, c_neg, k)
    
    # We have 20 positive samples.
    # The user wants graded margins [0, 0.1] applied to pairs.
    # We have 10 pairs in 20 samples. So margins: 0.01, 0.02, 0.03 ... 0.10
    margins = np.linspace(0.01, 0.10, 10)
    
    raw_advantages = np.zeros(N)
    colors = []
    labels = []
    
    # 1. Assign advantages to the 20 positive samples
    for i in range(10):  # 10 pairs
        margin = margins[i]
        raw_advantages[i*2] = a_pos_base + margin
        raw_advantages[i*2 + 1] = a_pos_base + margin
        colors.extend(['blue', 'blue'])
        labels.extend([f'+Margin {margin:.2f}', f'+Margin {margin:.2f}'])
        
    # 2. Assign advantages to the 12 negative samples
    for i in range(c_maj, N):
        raw_advantages[i] = a_neg_base
        colors.append('red')
        labels.append('Negative Sample')
        
    # 3. Apply Group Z-Score Normalization
    mean_group = np.mean(raw_advantages)
    std_group = np.std(raw_advantages) + 1e-8
    
    norm_advantages = (raw_advantages - mean_group) / std_group
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # X axis positions
    x_pos = np.arange(c_maj)
    x_neg = np.arange(c_maj, N)
    
    # ==========================================
    # Top Plot: RAW (Unnormalized) Advantages
    # ==========================================
    scatter_raw = ax1.scatter(x_pos, raw_advantages[:c_maj], 
                          c=margins.repeat(2), cmap='viridis', s=100, 
                          edgecolor='black', zorder=3, label='Positive (Correct)')
    
    ax1.scatter(x_neg, raw_advantages[c_maj:], 
                color='red', marker='X', s=100, 
                zorder=3, label='Negative (Incorrect)')
    
    # Add labels near positive samples to show margin
    for i in range(10):
        idx = i * 2
        ax1.annotate(f'+{margins[i]:.2f}', 
                     (x_pos[idx] + 0.5, raw_advantages[idx] + 0.01),
                     fontsize=9, ha='center', color='darkblue')

    ax1.axhline(mean_group, color='green', linestyle='-', alpha=0.6, label='Group Mean')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.6)
    
    ax1.set_title(f'Raw (Unnormalized) Advantage Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Raw Advantage Value', fontsize=12)
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Add text box with stats to ax1
    stats_text = (
        f"Raw Baseline Pos Adv: {a_pos_base:.4f}\n"
        f"Raw Baseline Neg Adv: {a_neg_base:.4f}\n"
        f"Group Mean: {mean_group:.4f}\n"
        f"Group Std: {std_group:.4f}"
    )
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ==========================================
    # Bottom Plot: NORMALIZED Advantages
    # ==========================================
    scatter_norm = ax2.scatter(x_pos, norm_advantages[:c_maj], 
                          c=margins.repeat(2), cmap='viridis', s=100, 
                          edgecolor='black', zorder=3, label='Positive (Correct)')
    
    ax2.scatter(x_neg, norm_advantages[c_maj:], 
                color='red', marker='X', s=100, 
                zorder=3, label='Negative (Incorrect)')
    
    # Add labels near positive samples to show margin
    for i in range(10):
        idx = i * 2
        ax2.annotate(f'+{margins[i]:.2f}', 
                     (x_pos[idx] + 0.5, norm_advantages[idx] + 0.05),
                     fontsize=9, ha='center', color='darkblue')

    ax2.axhline(0, color='black', linestyle='--', alpha=0.6)
    
    ax2.set_title(f'Normalized Advantage Distribution (Z-Score)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index (0-19: Positive with increasing margin, 20-31: Negative)', fontsize=14)
    ax2.set_ylabel('Normalized Advantage (Z-Score)', fontsize=12)
    
    ax2.legend(loc='lower left', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # Add a single colorbar for both
    cbar = fig.colorbar(scatter_norm, ax=[ax1, ax2], location='right')
    cbar.set_label('Added Margin Level', fontsize=12)

    plt.suptitle(f'Impact of Graded Margins on a Single Prompt Group ($N={N}, k={k}, c_{{maj}}={c_maj}$)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    save_path = 'd:\\Repository\\GitHub\\DIV-TTRL\\verl\\scripts\\graded_margin_single_group.png'
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_specific_sample_distribution(N=32, k=4, c_maj=20)
