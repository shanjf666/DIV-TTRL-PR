import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln

sns.set_theme(style="whitegrid")

def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def compute_pass_k_adv(N, c_neg, k):
    if c_neg >= k:
        log_prob_fail = log_comb(c_neg, k) - log_comb(N, k)
        prob_fail = np.exp(log_prob_fail)
    else:
        prob_fail = 0.0
        
    r_mean = 1.0 - prob_fail
    std = np.sqrt(r_mean * (1.0 - r_mean) + 1e-8)
    
    a_pos = prob_fail / std
    
    if c_neg >= 1 and c_neg - 1 >= k - 1:
        log_p_cond_fail = log_comb(c_neg - 1, k - 1) - log_comb(N - 1, k - 1)
        p_cond_fail = np.exp(log_p_cond_fail)
    else:
        p_cond_fail = 0.0
        
    a_neg = (prob_fail - p_cond_fail) / std
    return a_pos, a_neg

def plot_advantage(N=16, k=4, lam=0.05, c_max=2.0):
    sc_ratios = [0.2,0.4,0.6, 0.8, 1.0]
    num_trials = 50000
    z_eval = np.linspace(0, 3, 100) # Since it's symmetric around 0
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharey=True)
    
    for idx, sc in enumerate(sc_ratios):
        c_maj = int(N * sc)
        c_neg = N - c_maj
        
        a_pos, a_neg = compute_pass_k_adv(N, c_neg, k)
        
        # Approximate expected mu and sigma for the group after adding length term
        z_random = np.random.randn(num_trials, N)
        # Empirical normalize lengths in group
        z_random = (z_random - z_random.mean(axis=1, keepdims=True)) / (z_random.std(axis=1, keepdims=True) + 1e-8)
        
        A_prime = np.zeros((num_trials, N))
        A_prime[:, :c_maj] = a_pos + lam * np.minimum(np.abs(z_random[:, :c_maj]), c_max)
        if c_neg > 0:
            A_prime[:, c_maj:] = a_neg
            
        mu_A_prime = A_prime.mean(axis=1)
        sigma_A_prime = A_prime.std(axis=1) + 1e-8
        
        expected_mu = mu_A_prime.mean()
        expected_sigma = sigma_A_prime.mean()
        
        print(f"sc_ratio={sc:.1f} | c_maj={c_maj} | c_neg={c_neg}")
        print(f"  a_pos: {a_pos:.3f}, a_neg: {a_neg:.3f}")
        print(f"  New Mu: {expected_mu:.3f}, New Sigma: {expected_sigma:.3f}")
        
        # Calculate A_final for eval points
        A_prime_eval = a_pos + lam * np.minimum(np.abs(z_eval), c_max)
        A_final_eval = (A_prime_eval - expected_mu) / expected_sigma
        
        # Original re-normalized ? No, original is constant. Wait, if we DO NOT renormalize the original, 
        # it might have different mean/std than the new one. 
        # But wait, original a_pos and a_neg are mathematically derived.
        # What if we empirically normalize the original too?
        A_base = np.zeros((num_trials, N))
        A_base[:, :c_maj] = a_pos
        if c_neg > 0:
            A_base[:, c_maj:] = a_neg
        mu_A_base = A_base.mean(axis=1)
        sigma_A_base = A_base.std(axis=1) + 1e-8
        
        # Expected base normalized advantages
        A_base_norm_pos = ((a_pos - mu_A_base) / sigma_A_base).mean()
        A_base_norm_neg = ((a_neg - mu_A_base) / sigma_A_base).mean() if c_neg > 0 else 0
        
        ax = axes[idx]
        ax.plot(z_eval, A_final_eval, label='New Adv (Majority)', color='blue', lw=2)
        ax.axhline(a_pos, label='Raw Pass@k Adv (Majority)', color='blue', linestyle='--', alpha=0.6)
        
        # Show normalized base as well
        if sc < 1.0:
            ax.axhline(A_base_norm_pos, label='Renormalized Pass@k (Maj)', color='green', linestyle=':', alpha=0.6)
            
            A_final_min = (a_neg - expected_mu) / expected_sigma
            ax.axhline(A_final_min, label='New Adv (Minority)', color='red', lw=2)
            ax.axhline(a_neg, label='Raw Pass@k Adv (Minority)', color='red', linestyle='--', alpha=0.6)
            ax.axhline(A_base_norm_neg, label='Renormalized Pass@k (Min)', color='orange', linestyle=':', alpha=0.6)
            
        ax.set_title(f'sc_ratio = {sc:.2f} (c_{{maj}}={c_maj})')
        ax.set_xlabel('Length z-score $|l_i - \\mu_L| / \\sigma_L$')
        if idx == 0:
            ax.set_ylabel('Advantage Value')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('d:\\Repository\\GitHub\\DIV-TTRL\\verl\\scripts\\advantage_length_plot.png', dpi=150)
    print("Plot saved to d:\\Repository\\GitHub\\DIV-TTRL\\verl\\scripts\\advantage_length_plot.png")

if __name__ == "__main__":
    plot_advantage(N=16, k=4, lam=0.05, c_max=2.0)
