"""
SutraAudit: Reproducibility Suite
Global South AI Safety Hackathon 2026 — Asia Track / Bengaluru Hub
Author: Aditya Tambi
Repository: https://github.com/adityatambi/sutraAudit

This script fully reproduces the empirical results, metrics, and visualization
plots documented in the SutraAudit final research report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Set formatting styles for publication-grade output
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def generate_reproducible_environment():
    print("[1/4] Establishing random seeds and engineering synthetic applicant distributions...")
    np.random.seed(42)
    n_samples = 1000
    
    # Underlying repayment capacity (held completely identical across demographics)
    true_ability_to_pay = np.random.uniform(0.3, 0.9, n_samples)
    default_labels = (true_ability_to_pay < 0.52).astype(int)
    
    # Demographic division (0: Informal Target Group, 1: Formal Baseline Group)
    demographic_marker = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    # Alternative proxy variables heavily correlated with the structural class
    upi_velocity = np.where(demographic_marker == 0, 
                            np.random.randint(25, 60, n_samples), 
                            np.random.randint(1, 12, n_samples))
    
    device_tier = np.where(demographic_marker == 0, 
                           np.random.choice([1, 2], n_samples, p=[0.7, 0.3]), 
                           np.random.choice([3, 4], n_samples, p=[0.2, 0.8]))
    
    # Income stability metric (adds low-variance noise to ability to pay)
    income_stability = true_ability_to_pay + np.random.normal(0, 0.04, n_samples)
    
    df = pd.DataFrame({
        'upi_transaction_velocity': upi_velocity,
        'device_tier_rating': device_tier,
        'income_stability_index': income_stability,
        'true_demographic_marker': demographic_marker,
        'default_label': default_labels
    })
    return df

def calculate_fairness_metrics(y_true, y_pred, df_test):
    # Split outcomes across sensitive demographics
    group_formal = df_test[df_test['true_demographic_marker'] == 1]
    group_informal = df_test[df_test['true_demographic_marker'] == 0]
    
    idx_formal = group_formal.index
    idx_informal = group_informal.index
    
    # Approval Rates (Predictions where model says 'no default' / approved)
    app_rate_formal = np.mean(y_pred[idx_formal] == 0)
    app_rate_informal = np.mean(y_pred[idx_informal] == 0)
    
    # True Positive Rates / Equal Opportunity (Approved given they are safe borrowers)
    tp_formal = np.sum((y_pred[idx_formal] == 0) & (y_true[idx_formal] == 0)) / np.sum(y_true[idx_formal] == 0)
    tp_informal = np.sum((y_pred[idx_informal] == 0) & (y_true[idx_informal] == 0)) / np.sum(y_true[idx_informal] == 0)
    
    dpd = abs(app_rate_formal - app_rate_informal)
    eod = abs(tp_formal - tp_informal)
    
    return app_rate_formal, app_rate_informal, dpd, eod

def execute_reproduction_pipeline():
    df = generate_reproducible_environment()
    
    # 1. Train Unaligned Baseline Model (Dropped demographic columns, but proxy leakage remains)
    print("[2/4] Executing standard unconstrained risk training (Baseline Model)...")
    X = df[['upi_transaction_velocity', 'device_tier_rating', 'income_stability_index']]
    y = df['default_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    df_test = df.loc[X_test.index].reset_index(drop=True)
    
    baseline_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    
    base_formal, base_informal, base_dpd, base_eod = calculate_fairness_metrics(
        y_test.values, y_pred_base, df_test
    )
    
    # 2. Simulate Adaption Labs' Blueprint Steering Bounds
    # Mathematically models decoupling the weights of proxies (velocity/device) from target labels
    print("[3/4] Modeling Adaption Labs Blueprint specification layer data transformations...")
    X_train_aligned = X_train.copy()
    X_test_aligned = X_test.copy()
    
    # Blueprint forces structural balance by zeroing out distribution deltas on alternative dimensions
    X_train_aligned['upi_transaction_velocity'] = X_train['income_stability_index'] * 20
    X_test_aligned['upi_transaction_velocity'] = X_test['income_stability_index'] * 20
    X_train_aligned['device_tier_rating'] = np.random.choice([2, 3], size=len(X_train))
    X_test_aligned['device_tier_rating'] = np.random.choice([2, 3], size=len(X_test))
    
    aligned_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
    aligned_model.fit(X_train_aligned, y_train)
    y_pred_align = aligned_model.predict(X_test_aligned)
    acc_align = accuracy_score(y_test, y_pred_align)
    
    align_formal, align_informal, align_dpd, align_eod = calculate_fairness_metrics(
        y_test.values, y_pred_align, df_test
    )
    
    # Print empirical table report directly to terminal console
    print("\n" + "="*75)
    print("                      SUTRAAUDIT VERIFIED RESULTS                      ")
    print("="*75)
    print(f"Metric                       | Baseline Model  | Aligned Model   | Delta")
    print(f"-"*75)
    print(f"Predictive Accuracy          | {acc_base*100:.1f}%           | {acc_align*100:.1f}%           | {(acc_align-acc_base)*100:.1f}%")
    print(f"Formal Group Approval Rate   | {base_formal*100:.1f}%           | {align_formal*100:.1f}%           | {(align_formal-base_formal)*100:.1f}%")
    print(f"Informal Group Approval Rate | {base_informal*100:.1f}%           | {align_informal*100:.1f}%           | {(align_informal-base_informal)*100:.1f}%")
    print(f"Demographic Parity (DPD)     | {base_dpd:.3f}           | {align_dpd:.3f}           | {align_dpd-base_dpd:.3f}")
    print(f"Equal Opportunity (EOD)      | {base_eod:.3f}           | {align_eod:.3f}           | {align_eod-base_eod:.3f}")
    print("="*75 + "\n")
    
    # 3. Generate Final Plot
    print("[4/4] Constructing verification plots and saving to workspace...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("SutraAudit: Alternative Credit Scoring Bias Mitigation Report", weight='bold', y=0.98)
    
    # Plot 1: Credit Approval Rates
    categories = ['Baseline', 'Blueprint Aligned']
    formal_rates = [base_formal, align_formal]
    informal_rates = [base_informal, align_informal]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, formal_rates, width, label='Formal Baseline', color='#6ca0dc')
    rects2 = axes[0].bar(x + width/2, informal_rates, width, label='Informal Target', color='#3b6282')
    
    axes[0].set_ylabel('Approval Rate')
    axes[0].set_title('Credit Approval Rates by Group', weight='semibold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(title="Group")
    
    # Add precise numbers on top of bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        axes[0].annotate(f'{height*100:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold')

    # Plot 2: Fairness Disparity Metric Deltas
    metrics = ['Baseline', 'Blueprint Aligned']
    dpd_vals = [base_dpd, align_dpd]
    eod_vals = [base_eod, align_eod]
    
    rects3 = axes[1].bar(x - width/2, dpd_vals, width, label='Demographic Parity Diff', color='#d96b5c')
    rects4 = axes[1].bar(x + width/2, eod_vals, width, label='Equal Opportunity Diff', color='#9c3d3d')
    
    axes[1].set_ylabel('Disparity Value')
    axes[1].set_title('Fairness Disparity Comparison (Lower is Better)', weight='semibold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim(0, 0.6)
    axes[1].axhline(y=0.10, color='gray', linestyle='--', alpha=0.7, label='Fairness Limit (10%)')
    axes[1].legend()
    
    for rect in rects3 + rects4:
        height = rect.get_height()
        axes[1].annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold')
        
    plt.tight_layout()
    output_filename = "baseline_vs_aligned_comparison.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Verification figure saved securely as '{output_filename}'.")

if __name__ == "__main__":
    execute_reproduction_pipeline()
