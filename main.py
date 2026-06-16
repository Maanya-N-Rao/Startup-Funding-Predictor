# startup_predictor.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. Synthetic Indian Startup Dataset ───────────────────────────────────────
np.random.seed(42)
N = 3_000

SECTORS   = ['Fintech','Healthtech','Edtech','SaaS','D2C','Agritech','Deeptech','EV','Logistics']
STAGES    = ['Idea','Pre-Seed','Seed','Series A','Series B']
CITIES    = ['Bengaluru','Mumbai','Delhi','Hyderabad','Pune','Chennai']
ECELL_IDS = [True, False]

df = pd.DataFrame({
    'sector':            np.random.choice(SECTORS, N),
    'stage':             np.random.choice(STAGES, N, p=[0.1,0.25,0.30,0.25,0.10]),
    'city':              np.random.choice(CITIES, N),
    'founding_year':     np.random.randint(2010, 2024, N),
    'team_size':         np.random.poisson(8, N).clip(1, 100),
    'founders_count':    np.random.randint(1, 5, N),
    'iit_iim_founder':   np.random.choice([0, 1], N, p=[0.6, 0.4]),
    'prior_exits':       np.random.poisson(0.3, N).clip(0, 3),
    'patents':           np.random.poisson(0.8, N).clip(0, 10),
    'revenue_lakhs':     np.random.exponential(50, N),
    'monthly_burn_l':    np.random.exponential(20, N),
    'runway_months':     np.random.uniform(3, 36, N),
    'mrr_growth_pct':    np.random.normal(15, 20, N),
    'customer_count':    np.random.exponential(200, N).astype(int),
    'nps_score':         np.random.randint(-50, 80, N),
    'linkedin_followers':np.random.exponential(1000, N).astype(int),
    'ecell_portfolio':   np.random.choice(ECELL_IDS, N, p=[0.3, 0.7]),
    'kdem_registered':   np.random.choice([0, 1], N, p=[0.5, 0.5]),
    'prev_funding_cr':   np.random.exponential(2, N),
    'market_size_cr':    np.random.exponential(5000, N),
})

# Funding success probability
stage_score = {'Idea': 0, 'Pre-Seed': 1, 'Seed': 2, 'Series A': 3, 'Series B': 4}
df['stage_n'] = df['stage'].map(stage_score)
logit = (
    0.3 * df['stage_n'] +
    0.02 * df['iit_iim_founder'] +
    0.15 * df['prior_exits'] +
    0.01 * df['patents'] +
    0.003 * (df['mrr_growth_pct'].clip(-50,100)) +
    0.2 * df['ecell_portfolio'].astype(int) +
    0.15 * df['kdem_registered'] +
    0.1 * (df['runway_months'] > 12).astype(int) -
    0.005 * df['monthly_burn_l'] +
    np.random.randn(N) * 0.5 - 1.5
)
df['funded'] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
print(f"Dataset: {N} startups | Funded: {df['funded'].mean():.1%}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
le = LabelEncoder()
for col in ['sector', 'stage', 'city']:
    df[f'{col}_enc'] = le.fit_transform(df[col])

df['burn_multiple'] = df['monthly_burn_l'] / (df['revenue_lakhs'] / 12 + 0.01)
df['startup_age']   = 2025 - df['founding_year']
df['rev_per_emp']   = df['revenue_lakhs'] / df['team_size']

FEATURES = [
    'sector_enc','stage_enc','city_enc','startup_age','team_size','founders_count',
    'iit_iim_founder','prior_exits','patents','revenue_lakhs','monthly_burn_l',
    'runway_months','mrr_growth_pct','customer_count','nps_score','linkedin_followers',
    'ecell_portfolio','kdem_registered','prev_funding_cr','market_size_cr',
    'burn_multiple','stage_n','rev_per_emp'
]
X = df[FEATURES].astype(float)
y = df['funded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ── 3. Models ─────────────────────────────────────────────────────────────────
# Logistic Regression
pipe_lr = Pipeline([('scaler', StandardScaler()),
                    ('model', LogisticRegression(max_iter=1000, C=0.5, random_state=42))])
pipe_lr.fit(X_train, y_train)
lr_proba = pipe_lr.predict_proba(X_test)[:,1]
lr_auc   = roc_auc_score(y_test, lr_proba)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:,1]
rf_auc   = roc_auc_score(y_test, rf_proba)

# XGBoost + GridSearch
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8],
}
xgb_base = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc')
gs = GridSearchCV(xgb_base, param_grid, cv=StratifiedKFold(3),
                  scoring='roc_auc', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
best_xgb  = gs.best_estimator_
xgb_proba = best_xgb.predict_proba(X_test)[:,1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_pred  = best_xgb.predict(X_test)

print(f"\n📊 Model AUC Scores:")
print(f"   Logistic Regression : {lr_auc:.4f}")
print(f"   Random Forest       : {rf_auc:.4f}")
print(f"   XGBoost (tuned)     : {xgb_auc:.4f}")
print(f"\nBest XGB Params: {gs.best_params_}")
print(f"\n{classification_report(y_test, xgb_pred, target_names=['Not Funded','Funded'])}")

# ── 4. SHAP Analysis ──────────────────────────────────────────────────────────
explainer  = shap.TreeExplainer(best_xgb)
shap_vals  = explainer.shap_values(X_test)

# ── 5. Visualizations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Startup Funding Success Predictor', fontsize=16, fontweight='bold')

# ROC
for name, proba in [('LR', lr_proba), ('RF', rf_proba), ('XGBoost', xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0,0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
axes[0,0].plot([0,1],[0,1],'k--'); axes[0,0].set_title('ROC Curves')
axes[0,0].set_xlabel('FPR'); axes[0,0].set_ylabel('TPR'); axes[0,0].legend()

# SHAP Beeswarm (mean abs)
shap_mean = np.abs(shap_vals).mean(axis=0)
idx_sort  = np.argsort(shap_mean)[-12:]
axes[0,1].barh([FEATURES[i] for i in idx_sort], shap_mean[idx_sort], color='steelblue')
axes[0,1].set_title('SHAP Feature Importance (Mean |SHAP|)')

# Sector-wise funding rate
sec_rate = df.groupby('sector')['funded'].mean().sort_values()
axes[1,0].barh(sec_rate.index, sec_rate.values*100, color='steelblue', edgecolor='k')
axes[1,0].set_title('Funding Rate by Sector (%)')
axes[1,0].set_xlabel('% Funded')

# SHAP Waterfall (single prediction)
sample_idx = 0
shap_single = shap_vals[sample_idx]
top_n = 8
idx_top = np.argsort(np.abs(shap_single))[-top_n:]
colors  = ['red' if v < 0 else 'green' for v in shap_single[idx_top]]
axes[1,1].barh([FEATURES[i] for i in idx_top], shap_single[idx_top], color=colors)
axes[1,1].axvline(0, color='k', linewidth=0.8)
axes[1,1].set_title('SHAP Waterfall (Single Prediction)')
axes[1,1].set_xlabel('SHAP Value')

plt.tight_layout()
plt.savefig('startup_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved: startup_results.png")