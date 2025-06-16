import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ----------------------------
# 1. 任意数の群データ生成
# ----------------------------
np.random.seed(1)
groups = ['Control', 'A', 'B', 'C', 'D', 'E']  # 複数の処理群
n_per_group = 20

data = pd.DataFrame({
    'group': sum([[g] * n_per_group for g in groups], []),
    'value': np.concatenate([
        np.random.normal(5.0, 1, n_per_group),  # Control
        np.random.normal(6.0, 1, n_per_group),  # A
        np.random.normal(6.5, 1, n_per_group),  # B
        np.random.normal(5.2, 1, n_per_group),  # C
        np.random.normal(5.5, 1, n_per_group),  # D
        np.random.normal(4.3, 2, n_per_group),
    ])
})

# ----------------------------
# 2. ANOVA確認
# ----------------------------
model = ols('value ~ group', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA:")
print(anova_table)

# ----------------------------
# 3. t検定（Control vs 各群）+ 95%信頼区間
# ----------------------------
control = data[data['group'] == 'Control']['value']
results = []
df_resid = len(data) - len(groups)

for group in groups:
    if group == 'Control':
        continue
    treatment = data[data['group'] == group]['value']
    t_stat, p_val = ttest_ind(treatment, control, equal_var=True)
    
    # 平均差と信頼区間
    mean_diff = treatment.mean() - control.mean()
    pooled_sd = np.sqrt(((treatment.std(ddof=1) ** 2) + (control.std(ddof=1) ** 2)) / 2)
    se = pooled_sd * np.sqrt(1/n_per_group + 1/n_per_group)
    ci_range = t.ppf(0.975, df_resid) * se
    ci_lower = mean_diff - ci_range
    ci_upper = mean_diff + ci_range

    results.append({
        'Comparison': f'{group} vs Control',
        't_stat': t_stat,
        'p_raw': p_val,
        'Mean diff': mean_diff,
        'CI lower': ci_lower,
        'CI upper': ci_upper
    })

# ----------------------------
# 4. Holm補正（近似Dunnett法）
# ----------------------------
df_results = pd.DataFrame(results)
df_results['p_holm'] = multipletests(df_results['p_raw'], method='holm')[1]
df_results['Significant (p<0.05)'] = df_results['p_holm'] < 0.05

# ----------------------------
# 5. 結果出力
# ----------------------------
print("\nDunnett法（近似） with 95%信頼区間:")
print(df_results[['Comparison', 'Mean diff', 'CI lower', 'CI upper', 'p_raw', 'p_holm', 'Significant (p<0.05)']])
