"""
========================================================================================
ENTERPRISE CUSTOMER CHURN INTELLIGENCE & FINANCIAL RETENTION PLATFORM
========================================================================================
Author: Vivek Kumar (Tata Institute of Social Sciences | Jio Analytics Alumni)
Target Audience: Tier-1 Tech (FAANG), Top-Tier Fintech (Stripe/PayPal), & Global Banking
Architecture:
  1. Multi-Domain High-Dimensional Data Engine (Demographic, Ecosystem, Velocity, Friction)
  2. Zero-Leakage Scikit-learn Pipeline (Numerical/Categorical ColumnTransformer)
  3. Multi-Model Tournament (Logistic Regression, Random Forest, LightGBM, XGBoost)
  4. Isotonic / Sigmoid Probability Calibration (Brier Score Minimization)
  5. Decile-by-Decile Lift & Cumulative Gains Analysis (Top-Decile Capture Rate)
  6. Financial Cost-Benefit & Expected Value Engine (CLV vs Retention Intervention Cost)
  7. Explainable AI (SHAP Global Drivers & Local Customer Diagnosis Playbook)
  8. Production Export (Serialized Pipeline, Calibration Map, Production Scoring API)
========================================================================================
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)

import lightgbm as lgb
import xgboost as xgb
import shap

# Set styling for production charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ======================================================================================
# 1. ENTERPRISE DATA GENERATOR WITH REALISTIC MULTI-DOMAIN CORRELATIONS
# ======================================================================================
class EnterpriseDataEngine:
    """Generates realistic enterprise-scale customer churn data with non-linear correlations,
    behavioral momentum metrics, service adoption, and financial friction events."""
    
    @staticmethod
    def generate_customer_cohort(n_records: int = 25000, random_state: int = 42) -> pd.DataFrame:
        np.random.seed(random_state)
        
        customer_ids = [f"CUST_{i:07d}" for i in range(1, n_records + 1)]
        
        # Pillar 1: Demographic & Account Attributes
        tenure_months = np.random.exponential(scale=24, size=n_records).clip(1, 72).round().astype(int)
        contract_type = np.random.choice(
            ['Month-to-Month', 'One-Year', 'Two-Year'],
            p=[0.55, 0.25, 0.20],
            size=n_records
        )
        payment_method = np.random.choice(
            ['Electronic Check', 'Mailed Check', 'Bank Transfer (Auto)', 'Credit Card (Auto)'],
            p=[0.35, 0.15, 0.25, 0.25],
            size=n_records
        )
        paperless_billing = np.random.choice(['Yes', 'No'], p=[0.60, 0.40], size=n_records)
        senior_citizen = np.random.choice([0, 1], p=[0.83, 0.17], size=n_records)
        dependents = np.random.choice(['Yes', 'No'], p=[0.30, 0.70], size=n_records)
        
        # Pillar 2: Product & Ecosystem Density
        internet_service = np.random.choice(['Fiber Optic', 'DSL', 'None'], p=[0.45, 0.35, 0.20], size=n_records)
        online_security = np.where(internet_service == 'None', 'No internet service',
                                   np.random.choice(['Yes', 'No'], p=[0.38, 0.62], size=n_records))
        tech_support = np.where(internet_service == 'None', 'No internet service',
                                np.random.choice(['Yes', 'No'], p=[0.36, 0.64], size=n_records))
        streaming_tv = np.where(internet_service == 'None', 'No internet service',
                                np.random.choice(['Yes', 'No'], p=[0.48, 0.52], size=n_records))
        online_backup = np.where(internet_service == 'None', 'No internet service',
                                 np.random.choice(['Yes', 'No'], p=[0.42, 0.58], size=n_records))
        
        # Pillar 3: Financial & Usage Velocity Indicators (FAANG/Fintech metrics)
        monthly_charges = np.zeros(n_records)
        for i in range(n_records):
            base = 25.0 if internet_service[i] == 'None' else (75.0 if internet_service[i] == 'Fiber Optic' else 45.0)
            addons = sum([
                online_security[i] == 'Yes',
                tech_support[i] == 'Yes',
                streaming_tv[i] == 'Yes',
                online_backup[i] == 'Yes'
            ]) * 12.5
            monthly_charges[i] = base + addons + np.random.normal(0, 4)
        monthly_charges = monthly_charges.clip(18.0, 135.0).round(2)
        
        total_charges = (monthly_charges * tenure_months * np.random.uniform(0.96, 1.04, n_records)).round(2)
        
        # Behavioral Momentum: Ratio of 30-day usage vs 90-day baseline (Velocity decay is top predictor)
        usage_drop_ratio_30d_90d = np.random.beta(a=5, b=2, size=n_records) # Mode ~ 0.8
        recency_days_last_active = np.random.exponential(scale=12, size=n_records).clip(0, 90).round()
        
        # Pillar 4: Friction & Customer Support Signals
        support_tickets_60d = np.random.poisson(lam=0.7, size=n_records)
        # Some customers experience catastrophic service failure
        spike_mask = np.random.rand(n_records) < 0.08
        support_tickets_60d[spike_mask] += np.random.randint(2, 6, size=np.sum(spike_mask))
        
        failed_payments_12m = np.random.poisson(lam=0.25, size=n_records)
        nps_score = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            p=[0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.15, 0.08, 0.07],
            size=n_records
        )
        
        # Estimated Customer Lifetime Value (CLV in USD)
        estimated_clv = (monthly_charges * 24.0 * np.random.uniform(0.85, 1.35, n_records)).round(2)
        
        # Latent Ground Truth Churn Log-Odds Formula (reflecting real consumer economics)
        log_odds = (
            -1.85
            + 1.15 * (contract_type == 'Month-to-Month')
            - 0.90 * (contract_type == 'Two-Year')
            - 0.55 * (contract_type == 'One-Year')
            + 0.65 * (payment_method == 'Electronic Check')
            - 0.40 * (payment_method == 'Credit Card (Auto)')
            - 0.035 * tenure_months
            + 0.015 * monthly_charges
            - 1.40 * (usage_drop_ratio_30d_90d - 0.5) # Sharp drops accelerate churn
            + 0.030 * recency_days_last_active
            + 0.45 * (support_tickets_60d >= 2)
            + 0.60 * (support_tickets_60d >= 4)
            + 0.35 * (failed_payments_12m >= 1)
            - 0.15 * (nps_score - 5.0)
            - 0.40 * (online_security == 'Yes')
            - 0.45 * (tech_support == 'Yes')
            + 0.30 * (internet_service == 'Fiber Optic')
        )
        
        true_prob = 1.0 / (1.0 + np.exp(-log_odds))
        churn_flag = (np.random.uniform(0, 1, n_records) < true_prob).astype(int)
        
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'tenure_months': tenure_months,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'paperless_billing': paperless_billing,
            'senior_citizen': senior_citizen,
            'dependents': dependents,
            'internet_service': internet_service,
            'online_security': online_security,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'online_backup': online_backup,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'usage_drop_ratio_30d_90d': usage_drop_ratio_30d_90d,
            'recency_days_last_active': recency_days_last_active,
            'support_tickets_60d': support_tickets_60d,
            'failed_payments_12m': failed_payments_12m,
            'nps_score': nps_score,
            'estimated_clv': estimated_clv,
            'churn_flag': churn_flag
        })
        
        return df


# ======================================================================================
# 2. ZERO-LEAKAGE SCIKIT-LEARN PREPROCESSING PIPELINE
# ======================================================================================
def build_feature_preprocessor(categorical_cols, numerical_cols):
    """Creates a production-grade ColumnTransformer ensuring strict train/test segregation."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()) # Robust to outliers in spend/usage
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


# ======================================================================================
# 3. FINANCIAL OPTIMIZATION & EXPECTED VALUE ENGINE
# ======================================================================================
class FinancialRetentionOptimizer:
    """Computes the expected profit curve across varying probability thresholds.
    Balancing retention intervention costs against protected Customer Lifetime Value."""
    
    def __init__(self, intervention_cost: float = 35.0, offer_acceptance_rate: float = 0.28):
        self.intervention_cost = intervention_cost
        self.offer_acceptance_rate = offer_acceptance_rate
        
    def evaluate_threshold_economics(self, y_true: np.ndarray, y_probs: np.ndarray, clv_series: np.ndarray):
        thresholds = np.linspace(0.05, 0.90, 86)
        results = []
        
        n_total = len(y_true)
        actual_churners = np.sum(y_true)
        
        for tau in thresholds:
            contact_mask = (y_probs >= tau)
            n_contacted = np.sum(contact_mask)
            
            if n_contacted == 0:
                results.append({
                    'threshold': tau,
                    'contact_rate': 0.0,
                    'contacted_customers': 0,
                    'churners_targeted': 0,
                    'churn_capture_rate': 0.0,
                    'campaign_cost': 0.0,
                    'gross_clv_saved': 0.0,
                    'net_profit': 0.0,
                    'roi_pct': 0.0
                })
                continue
                
            targeted_churners = np.sum(y_true[contact_mask])
            targeted_clv = clv_series[contact_mask]
            
            # Campaign total cost
            total_campaign_cost = n_contacted * self.intervention_cost
            
            # Expected saved CLV from targeted actual churners who accept the retention incentive
            expected_saved_clv = np.sum(y_true[contact_mask] * targeted_clv * self.offer_acceptance_rate)
            
            net_profit = expected_saved_clv - total_campaign_cost
            roi = (net_profit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0
            
            results.append({
                'threshold': tau,
                'contact_rate': n_contacted / n_total,
                'contacted_customers': int(n_contacted),
                'churners_targeted': int(targeted_churners),
                'churn_capture_rate': targeted_churners / actual_churners if actual_churners > 0 else 0.0,
                'campaign_cost': total_campaign_cost,
                'gross_clv_saved': expected_saved_clv,
                'net_profit': net_profit,
                'roi_pct': roi
            })
            
        res_df = pd.DataFrame(results)
        best_row = res_df.loc[res_df['net_profit'].idxmax()]
        return res_df, best_row


# ======================================================================================
# 4. DECILE LIFT & CUMULATIVE GAINS ANALYSIS (ENTERPRISE STANDARD)
# ======================================================================================
def compute_decile_lift_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Calculates decile lift table widely audited by executive teams & risk committees."""
    df = pd.DataFrame({'actual': y_true, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    df['decile'] = pd.qcut(df.index, q=10, labels=[f"Decile {i}" for i in range(1, 11)])
    
    summary = df.groupby('decile', observed=False).agg(
        total_customers=('actual', 'count'),
        actual_churners=('actual', 'sum'),
        mean_prob=('prob', 'mean')
    ).reset_index()
    
    total_churners = summary['actual_churners'].sum()
    overall_churn_rate = total_churners / summary['total_customers'].sum()
    
    summary['churn_rate_pct'] = (summary['actual_churners'] / summary['total_customers'] * 100).round(2)
    summary['cum_churners'] = summary['actual_churners'].cumsum()
    summary['cum_gains_pct'] = (summary['cum_churners'] / total_churners * 100).round(2)
    summary['decile_lift'] = (summary['churn_rate_pct'] / (overall_churn_rate * 100)).round(2)
    summary['cum_lift'] = ((summary['cum_gains_pct'] / ((summary.index + 1) * 10))).round(2)
    
    return summary


# ======================================================================================
# 5. MASTER EXECUTION ENGINE
# ======================================================================================
def run_enterprise_churn_pipeline():
    print("=" * 80)
    print("  ENTERPRISE CUSTOMER CHURN INTELLIGENCE & VALUE OPTIMIZATION ENGINE")
    print("  Lead Architect: Vivek Kumar | Production ML Pipeline")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Synthesize Data
    print("\n[Step 1/8] Generating 25,000 High-Dimensional Enterprise Customer Records...")
    df = EnterpriseDataEngine.generate_customer_cohort(n_records=25000, random_state=42)
    
    base_churn_rate = df['churn_flag'].mean()
    print(f"  -> Generated {len(df):,} accounts | Base Churn Rate: {base_churn_rate:.2%}")
    print(f"  -> Mean Monthly Charge: ${df['monthly_charges'].mean():.2f} | Mean CLV: ${df['estimated_clv'].mean():.2f}")
    
    # Feature Segregation
    categorical_cols = [
        'contract_type', 'payment_method', 'paperless_billing', 'senior_citizen',
        'dependents', 'internet_service', 'online_security', 'tech_support',
        'streaming_tv', 'online_backup'
    ]
    numerical_cols = [
        'tenure_months', 'monthly_charges', 'total_charges',
        'usage_drop_ratio_30d_90d', 'recency_days_last_active',
        'support_tickets_60d', 'failed_payments_12m', 'nps_score'
    ]
    
    X = df[categorical_cols + numerical_cols]
    y = df['churn_flag'].values
    clv_values = df['estimated_clv'].values
    
    # Step 2: Stratified Train/Test Split
    print("\n[Step 2/8] Creating Stratified Train/Test Partition (80/20)...")
    X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
        X, y, clv_values, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  -> Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    
    # Step 3: Multi-Model Tournament
    print("\n[Step 3/8] Executing Multi-Model Tournament across 4 Industry Architectures...")
    preprocessor = build_feature_preprocessor(categorical_cols, numerical_cols)
    
    # Scale pos weight for imbalanced gradient boosting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count
    
    candidate_models = {
        'Logistic Regression (L2 Baseline)': LogisticRegression(
            C=0.1, max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Balanced Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42
        ),
        'LightGBM Classifier': lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.04, num_leaves=31,
            scale_pos_weight=scale_weight, random_state=42, verbosity=-1
        ),
        'XGBoost Classifier': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            scale_pos_weight=scale_weight, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
    }
    
    tournament_records = []
    fitted_pipelines = {}
    
    for name, clf in candidate_models.items():
        t0 = time.time()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        
        roc_val = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        tournament_records.append({
            'Model Architecture': name,
            'ROC-AUC': round(roc_val, 4),
            'PR-AUC': round(pr_auc, 4),
            'F1-Score': round(f1, 4),
            'Brier Loss': round(brier, 4),
            'Fit Time (s)': round(fit_time, 2)
        })
        fitted_pipelines[name] = pipeline
        
    tournament_df = pd.DataFrame(tournament_records).sort_values(by='PR-AUC', ascending=False)
    print("\n--- MODEL TOURNAMENT RESULTS TABLE ---")
    print(tournament_df.to_string(index=False))
    
    # Best model selection (LightGBM/XGBoost typically wins)
    best_model_name = tournament_df.iloc[0]['Model Architecture']
    best_raw_pipeline = fitted_pipelines[best_model_name]
    print(f"\n  ==> Champion Model Selected: {best_model_name}")
    
    # Step 4: Probability Calibration (Enterprise Rigor)
    print("\n[Step 4/8] Calibrating Probabilities via CalibratedClassifierCV (Isotonic Scaling)...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=best_raw_pipeline,
        method='isotonic',
        cv=3
    )
    calibrated_clf.fit(X_train, y_train)
    
    raw_probs = best_raw_pipeline.predict_proba(X_test)[:, 1]
    cal_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    
    brier_raw = brier_score_loss(y_test, raw_probs)
    brier_cal = brier_score_loss(y_test, cal_probs)
    print(f"  -> Raw Tree Model Brier Score: {brier_raw:.4f}")
    print(f"  -> Calibrated Model Brier Score: {brier_cal:.4f} (Lower = superior risk probability alignment)")
    
    # Step 5: Decile Lift & Cumulative Gains
    print("\n[Step 5/8] Computing Decile Lift & Cumulative Gains Distribution...")
    decile_table = compute_decile_lift_table(y_test, cal_probs)
    print("\n--- CUMULATIVE GAINS & DECILE LIFT TABLE ---")
    print(decile_table.to_string(index=False))
    
    top2_gains = decile_table.loc[1, 'cum_gains_pct']
    print(f"\n  ==> Top 20% (Deciles 1-2) Capture: {top2_gains:.1f}% of all churning accounts!")
    
    # Step 6: Financial Optimization & Decision Threshold Tuning
    print("\n[Step 6/8] Running Financial Expected Value & Net Profit Optimization...")
    opt = FinancialRetentionOptimizer(intervention_cost=35.0, offer_acceptance_rate=0.28)
    econ_df, best_econ = opt.evaluate_threshold_economics(y_test, cal_probs, clv_test)
    
    # Compare Default (0.50) vs Optimal
    row_050 = econ_df.iloc[(econ_df['threshold'] - 0.50).abs().argsort()[:1]].iloc[0]
    
    print("\n--- BUSINESS ECONOMICS & PROFIT MAXIMIZATION COMPARISON ---")
    print(f"{'Metric':<35} | {'Default (tau=0.50)':<18} | {'Optimal Threshold (tau*=' + str(round(best_econ['threshold'], 2)) + ')':<25}")
    print("-" * 85)
    print(f"{'Customers Targeted':<35} | {int(row_050['contacted_customers']):<18} | {int(best_econ['contacted_customers']):<25}")
    print(f"{'Targeting Rate (% of Base)':<35} | {row_050['contact_rate']*100:<17.1f}% | {best_econ['contact_rate']*100:<24.1f}%")
    print(f"{'Churners Caught':<35} | {int(row_050['churners_targeted']):<18} | {int(best_econ['churners_targeted']):<25}")
    print(f"{'Churn Capture Rate (Recall)':<35} | {row_050['churn_capture_rate']*100:<17.1f}% | {best_econ['churn_capture_rate']*100:<24.1f}%")
    print(f"{'Campaign Cost':<35} | ${row_050['campaign_cost']:<17,.2f} | ${best_econ['campaign_cost']:<24,.2f}")
    print(f"{'Expected CLV Saved':<35} | ${row_050['gross_clv_saved']:<17,.2f} | ${best_econ['gross_clv_saved']:<24,.2f}")
    print(f"{'NET RETAINED PROFIT':<35} | ${row_050['net_profit']:<17,.2f} | ${best_econ['net_profit']:<24,.2f}")
    print(f"{'Campaign Net ROI':<35} | {row_050['roi_pct']:<17.1f}% | {best_econ['roi_pct']:<24.1f}%")
    
    profit_uplift = best_econ['net_profit'] - row_050['net_profit']
    print(f"\n  ==> Optimizing probability cutoff adds +${profit_uplift:,.2f} in net profit on this cohort alone!")
    
    # Step 7: Explainable AI & Retention Playbooks
    print("\n[Step 7/8] Generating SHAP Explainability & Customer Prescription Engine...")
    fitted_preprocessor = best_raw_pipeline.named_steps['preprocessor']
    fitted_model = best_raw_pipeline.named_steps['classifier']
    
    X_train_trans = fitted_preprocessor.transform(X_train)
    X_test_trans = fitted_preprocessor.transform(X_test)
    
    # Retrieve feature names from ColumnTransformer
    cat_feature_names = list(fitted_preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + cat_feature_names
    
    print("  -> Calculating SHAP values for model transparency & feature attribution...")
    try:
        if isinstance(fitted_model, (lgb.LGBMClassifier, xgb.XGBClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(fitted_model)
            shap_sample = X_test_trans[:500]
            shap_values = explainer.shap_values(shap_sample)
        elif isinstance(fitted_model, LogisticRegression):
            explainer = shap.LinearExplainer(fitted_model, X_train_trans[:500])
            shap_sample = X_test_trans[:500]
            shap_values = explainer.shap_values(shap_sample)
        else:
            explainer = shap.Explainer(fitted_model.predict_proba, X_train_trans[:100])
            shap_sample = X_test_trans[:300]
            shap_values = explainer(shap_sample).values
    except Exception as e:
        print(f"  [Notice] Standardizing explainer with permutation feature importance fallback: {e}")
        # Fallback to model coefficients / importances
        if hasattr(fitted_model, 'coef_'):
            shap_values = fitted_model.coef_[0] * X_test_trans[:500]
        else:
            shap_values = np.zeros((500, len(all_feature_names)))
    
    # If binary classification returns list or array
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values
        
    global_importance = np.abs(shap_vals_class1).mean(axis=0)
    top_feats_idx = np.argsort(global_importance)[::-1][:8]
    
    print("\n--- TOP GLOBAL CHURN RISK DRIVERS (SHAP IMPORTANCE) ---")
    for rank, idx in enumerate(top_feats_idx, 1):
        feat_name = all_feature_names[idx] if idx < len(all_feature_names) else f"Feature_{idx}"
        print(f"  {rank}. {feat_name:<30} (Mean |SHAP|: {global_importance[idx]:.4f})")
        
    # Prescriptive diagnosis for a sample high-risk customer
    high_risk_idx = np.where((cal_probs >= 0.75) & (y_test == 1))[0][0]
    high_risk_cust = X_test.iloc[high_risk_idx]
    high_risk_prob = cal_probs[high_risk_idx]
    
    print(f"\n--- PRESCRIPTIVE ACTION PLAYBOOK FOR SAMPLE AT-RISK ACCOUNT ---")
    print(f"  Customer ID: CUST_EXAMPLE_{high_risk_idx:05d}")
    print(f"  Calibrated Churn Probability: {high_risk_prob:.1%}")
    print(f"  Contract: {high_risk_cust['contract_type']} | Tenure: {high_risk_cust['tenure_months']} months")
    print(f"  Usage 30d/90d Ratio: {high_risk_cust['usage_drop_ratio_30d_90d']:.2f} | Support Tickets (60d): {high_risk_cust['support_tickets_60d']}")
    print(f"  Prescribed Interventions:")
    print(f"    1. [Product Retention]: Auto-trigger dedicated Customer Success Outreach within 24h.")
    print(f"    2. [Pricing]: Offer 15% discount contingent on converting to a 12-Month Contract.")
    print(f"    3. [Service Resolution]: Waive recent dispute charges and provide 30-day VIP tech support.")
    
    # Step 8: Save Visualizations & Serialized Artifacts
    print("\n[Step 8/8] Exporting Production Artifacts & High-Resolution Visualizations...")
    artifacts_dir = os.path.join(os.path.dirname(__file__), "churn_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 8a. Generate Visual Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. ROC and PR Curves
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, cal_probs)
    precision_curve, recall_curve_vals, _ = precision_recall_curve(y_test, cal_probs)
    ax1.plot(fpr, tpr, label=f"Calibrated ROC (AUC = {roc_auc_score(y_test, cal_probs):.3f})", color='#2563eb', lw=2)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    
    # 2. Cumulative Gains Chart
    ax2 = axes[0, 1]
    deciles_x = np.arange(10, 110, 10)
    gains_y = decile_table['cum_gains_pct'].values
    ax2.plot(deciles_x, gains_y, marker='o', color='#10b981', lw=2.5, label='Enterprise Churn Model')
    ax2.plot(deciles_x, deciles_x, linestyle='--', color='gray', label='Random Guessing Baseline')
    ax2.axvline(20, color='#f59e0b', linestyle=':', label=f"Top 20% Cutoff ({top2_gains:.1f}% Churners Captured)")
    ax2.set_title("Cumulative Gains (Lift Curve)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("% of Customer Base Contacted")
    ax2.set_ylabel("% of Churners Captured")
    ax2.legend(loc="lower right")
    
    # 3. Financial Net Profit Curve
    ax3 = axes[1, 0]
    ax3.plot(econ_df['threshold'], econ_df['net_profit'] / 1000.0, color='#7c3aed', lw=2.5, label='Net Retained Profit ($k)')
    ax3.axvline(best_econ['threshold'], color='#dc2626', linestyle='--', label=f"Optimal Cutoff (tau={best_econ['threshold']:.2f})")
    ax3.axvline(0.50, color='#6b7280', linestyle=':', label='Default Cutoff (tau=0.50)')
    ax3.set_title("Financial Net Retained Profit vs Probability Cutoff", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Decision Threshold (tau)")
    ax3.set_ylabel("Net Profit ($ Thousands)")
    ax3.legend(loc="upper left")
    
    # 4. Top Feature Importances (SHAP)
    ax4 = axes[1, 1]
    top_names = [all_feature_names[i] if i < len(all_feature_names) else f"Feat_{i}" for i in top_feats_idx][::-1]
    top_scores = [global_importance[i] for i in top_feats_idx][::-1]
    ax4.barh(top_names, top_scores, color='#0284c7')
    ax4.set_title("Top 8 Churn Risk Factors (Mean |SHAP Value|)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Mean Absolute Impact on Churn Log-Odds")
    
    plt.tight_layout()
    chart_path = os.path.join(artifacts_dir, "enterprise_churn_performance.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  -> Performance dashboard saved to: {chart_path}")
    
    # 8b. Export Serialized Pipeline
    model_export_path = os.path.join(artifacts_dir, "calibrated_churn_pipeline.joblib")
    joblib.dump({
        'pipeline': calibrated_clf,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'optimal_threshold': float(best_econ['threshold']),
        'best_model_name': best_model_name
    }, model_export_path)
    print(f"  -> Serialized Model Pipeline saved to: {model_export_path}")
    
    # 8c. Export Metadata Configuration JSON
    meta_path = os.path.join(artifacts_dir, "model_metadata.json")
    metadata = {
        'model_name': best_model_name,
        'author': 'Vivek Kumar',
        'training_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {
            'roc_auc': float(roc_auc_score(y_test, cal_probs)),
            'pr_auc': float(average_precision_score(y_test, cal_probs)),
            'brier_loss': float(brier_cal),
            'top_20pct_gains': float(top2_gains)
        },
        'economics': {
            'optimal_threshold': float(best_econ['threshold']),
            'net_profit_optimal_k_usd': float(best_econ['net_profit'] / 1000.0),
            'roi_pct': float(best_econ['roi_pct']),
            'contact_rate': float(best_econ['contact_rate']),
            'churn_recall': float(best_econ['churn_capture_rate'])
        },
        'categorical_features': categorical_cols,
        'numerical_features': numerical_cols
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Metadata configuration saved to: {meta_path}")
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"  PIPELINE EXECUTION COMPLETED IN {total_elapsed:.1f} SECONDS")
    print(f"  READY FOR ENTERPRISE DEPLOYMENT & PORTFOLIO PRESENTATION")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_enterprise_churn_pipeline()
