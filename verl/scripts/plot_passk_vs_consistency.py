# Note: When c_neg < k, it's impossible to pick a k-subset without a correct answer.
# P(fail) becomes 0, and the reward is saturated at 1.0. 
# This leads to a zero advantage for all samples in the Group.

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
    """计算原始的 Pass@k 优势（未进行 Group 归一化）"""
    # 总体失败概率 P(fail) = C(c_neg, k) / C(N, k)
    log_p_fail = log_comb(c_neg, k) - log_comb(N, k)
    p_fail = np.exp(log_p_fail) if log_p_fail > -np.inf else 0.0
    
    r_mean = 1.0 - p_fail
    # 这里使用 Bernoulli 标准差作为缩放因子（原文定义）
    sigma_b = np.sqrt(r_mean * (1.0 - r_mean) + 1e-8)
    
    # a_pos
    a_pos = p_fail / sigma_b
    
    # a_neg = (P_fail - P_cond_fail) / sigma_b
    log_p_cond_fail = log_comb(c_neg - 1, k - 1) - log_comb(N - 1, k - 1)
    p_cond_fail = np.exp(log_p_cond_fail) if log_p_cond_fail > -np.inf else 0.0
    a_neg = (p_fail - p_cond_fail) / sigma_b
    
    return a_pos, a_neg

def plot_normalized_passk_vs_consistency(N=32, k=4):
    c_maj_list = np.arange(1, N) # 不包含 N，因为 N 时 std 为 0
    sc_ratio_list = c_maj_list / N
    
    norm_a_pos_list = []
    norm_a_neg_list = []
    raw_a_pos_list = []
    
    for c_maj in c_maj_list:
        c_neg = N - c_maj
        a_pos, a_neg = compute_pass_k_adv_raw(N, c_neg, k)
        
        # 构造一个 Group 数组
        group_adv = np.array([a_pos] * c_maj + [a_neg] * c_neg)
        
        # 进行 Group 维度的归一化
        mu = np.mean(group_adv)
        std = np.std(group_adv)
        
        if std < 1e-8:
            norm_a_pos = 0.0
            norm_a_neg = 0.0
        else:
            norm_a_pos = (a_pos - mu) / std
            norm_a_neg = (a_neg - mu) / std
            
        norm_a_pos_list.append(norm_a_pos)
        norm_a_neg_list.append(norm_a_neg)
        raw_a_pos_list.append(a_pos)

    plt.figure(figsize=(10, 6))
    
    # 归一化后的曲线
    plt.plot(sc_ratio_list, norm_a_pos_list, marker='o', label='Normalized Majority Adv', color='blue', linewidth=2)
    plt.plot(sc_ratio_list, norm_a_neg_list, marker='x', linestyle='--', label='Normalized Minority Adv', color='red', linewidth=2)
    
    # 为了对比，画出原始 a_pos 的趋势（缩放后）
    plt.plot(sc_ratio_list, np.array(raw_a_pos_list), label='Raw Pass@k a_pos (Reference)', color='gray', alpha=0.3, linestyle=':')

    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'Group Normalized Pass@{k} Advantage (N={N})')
    plt.xlabel('Consistency Ratio (c_maj / N)')
    plt.ylabel('Advantage Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'd:\\Repository\\GitHub\\DIV-TTRL\\verl\\scripts\\passk_norm_advantage_curve.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_normalized_passk_vs_consistency(N=32, k=4)
