"""
========================================================================================
PRODUCTION CHURN INFERENCE SERVICE & RETENTION RECOMMENDATION API
========================================================================================
Author: Vivek Kumar
Demonstrates real-time customer risk scoring, risk tier assignment, and automated
prescriptive retention action triggers for CRM / relationship managers.
========================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "churn_artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "calibrated_churn_pipeline.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")


class ChurnInferenceEngine:
    """Production scoring engine loading calibrated model artifacts."""
    
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run pipeline first.")
            
        bundle = joblib.load(MODEL_PATH)
        self.pipeline = bundle['pipeline']
        self.categorical_cols = bundle['categorical_cols']
        self.numerical_cols = bundle['numerical_cols']
        self.optimal_threshold = bundle.get('optimal_threshold', 0.28)
        
        with open(META_PATH, 'r') as f:
            self.metadata = json.load(f)
            
    def score_customer(self, customer_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Scores a single customer record and returns actionable diagnosis."""
        df_single = pd.DataFrame([customer_payload])
        churn_prob = float(self.pipeline.predict_proba(df_single)[:, 1][0])
        
        # Risk Stratification
        if churn_prob >= 0.70:
            risk_tier = "CRITICAL RISK (Priority Tier 1)"
        elif churn_prob >= 0.45:
            risk_tier = "HIGH RISK (Priority Tier 2)"
        elif churn_prob >= 0.25:
            risk_tier = "MEDIUM RISK (Monitor)"
        else:
            risk_tier = "LOW RISK (Healthy Account)"
            
        is_above_optimal = churn_prob >= self.optimal_threshold
        
        # Prescriptive Recommendation Engine
        interventions = []
        if customer_payload.get('contract_type') == 'Month-to-Month':
            interventions.append("Offer 15% discount on upgrade to 1-Year or 2-Year Contract.")
        if customer_payload.get('usage_drop_ratio_30d_90d', 1.0) < 0.65:
            interventions.append(f"Usage down by {(1 - customer_payload.get('usage_drop_ratio_30d_90d'))*100:.0f}%: Trigger proactive Customer Success engagement.")
        if customer_payload.get('support_tickets_60d', 0) >= 2:
            interventions.append("High friction: Escalate to Senior Retention Specialist and offer 30-day VIP Support.")
        if customer_payload.get('payment_method') == 'Electronic Check':
            interventions.append("Incentivize enrollment in Credit Card / Auto-Pay with a $15 bill credit.")
        if not interventions:
            interventions.append("Maintain standard engagement cadence; account healthy.")
            
        clv = customer_payload.get('estimated_clv', customer_payload.get('monthly_charges', 70) * 24)
        expected_saved_clv = round(churn_prob * 0.28 * clv, 2)
        
        return {
            'customer_id': customer_payload.get('customer_id', 'UNKNOWN'),
            'calibrated_churn_probability': round(churn_prob, 4),
            'risk_tier': risk_tier,
            'campaign_targeting_recommended': is_above_optimal,
            'estimated_clv_usd': clv,
            'expected_saved_clv_usd': expected_saved_clv,
            'recommended_retention_actions': interventions
        }
        
    def batch_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring for customer base."""
        probs = self.pipeline.predict_proba(df)[:, 1]
        df_out = df.copy()
        df_out['calibrated_churn_probability'] = probs.round(4)
        df_out['actionable_target'] = (probs >= self.optimal_threshold).astype(int)
        df_out['risk_tier'] = pd.cut(
            probs,
            bins=[-0.01, 0.25, 0.45, 0.70, 1.01],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        return df_out


if __name__ == '__main__':
    print("=" * 70)
    print("  PRODUCTION CHURN INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    engine = ChurnInferenceEngine()
    
    # Test Customer 1: At-Risk customer with usage drop & month-to-month
    sample_risk = {
        'customer_id': 'CUST_9918234',
        'tenure_months': 4,
        'contract_type': 'Month-to-Month',
        'payment_method': 'Electronic Check',
        'paperless_billing': 'Yes',
        'senior_citizen': 0,
        'dependents': 'No',
        'internet_service': 'Fiber Optic',
        'online_security': 'No',
        'tech_support': 'No',
        'streaming_tv': 'Yes',
        'online_backup': 'No',
        'monthly_charges': 89.50,
        'total_charges': 358.00,
        'usage_drop_ratio_30d_90d': 0.42,
        'recency_days_last_active': 26,
        'support_tickets_60d': 3,
        'failed_payments_12m': 1,
        'nps_score': 3,
        'estimated_clv': 2148.00
    }
    
    # Test Customer 2: Highly loyal, 2-year contract customer
    sample_loyal = {
        'customer_id': 'CUST_1029384',
        'tenure_months': 48,
        'contract_type': 'Two-Year',
        'payment_method': 'Credit Card (Auto)',
        'paperless_billing': 'Yes',
        'senior_citizen': 0,
        'dependents': 'Yes',
        'internet_service': 'Fiber Optic',
        'online_security': 'Yes',
        'tech_support': 'Yes',
        'streaming_tv': 'Yes',
        'online_backup': 'Yes',
        'monthly_charges': 115.00,
        'total_charges': 5520.00,
        'usage_drop_ratio_30d_90d': 0.95,
        'recency_days_last_active': 2,
        'support_tickets_60d': 0,
        'failed_payments_12m': 0,
        'nps_score': 9,
        'estimated_clv': 3105.00
    }
    
    print("\n[Result 1] High-Risk Customer Scoring Output:")
    res1 = engine.score_customer(sample_risk)
    print(json.dumps(res1, indent=2))
    
    print("\n[Result 2] Loyal Customer Scoring Output:")
    res2 = engine.score_customer(sample_loyal)
    print(json.dumps(res2, indent=2))
    print("\n" + "=" * 70)
