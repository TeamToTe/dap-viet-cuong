"""
Downstream Utility Metric Plugin (Train on Synthetic, Test on Real - TSTR).
Evaluates whether augmenting with synthetic anomalies improves detection of real defects
using downstream classifiers (RandomForest, XGBoost/GradientBoosting, etc.).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from ..core.registry import register_metric
from ..core.base_metric import BaseMetric


@register_metric("utility")
class DownstreamUtilityMetric(BaseMetric):
    """
    Downstream utility evaluation:
    Trains classifiers on (Train Normal + Synthetic Anomalies),
    and evaluates on held-out Real Test data (Real Normal + Real Anomalies).
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _get_optimal_f1(self, y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
        best_f1, best_th = 0.0, 0.5
        for th in np.linspace(0.05, 0.95, 91):
            f = f1_score(y_true, (probs >= th).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = float(f)
                best_th = float(th)
        return best_f1, best_th

    def evaluate(
        self,
        real_norm: np.ndarray,
        real_anom: np.ndarray,
        synth_anom: np.ndarray,
        feature_names: List[str],
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        model_name: str = "SyntheticModel",
        **kwargs
    ) -> Dict[str, Any]:
        if X_test is None or y_test is None:
            raise ValueError("DownstreamUtilityMetric requires X_test and y_test in kwargs.")

        # Construct training set: Real Normal + Synthetic Anomalies
        X_tr = np.vstack([real_norm, synth_anom])
        y_tr = np.concatenate([np.zeros(len(real_norm)), np.ones(len(synth_anom))])
        perm = np.random.RandomState(self.random_state).permutation(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        # Classifiers to benchmark
        classifiers = [
            ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state))
        ]

        # Try XGBoost if installed
        try:
            import xgboost as xgb
            classifiers.append((
                "XGBoost",
                xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state, eval_metric="logloss", n_jobs=-1)
            ))
        except ImportError:
            pass

        results = []
        for clf_name, clf in classifiers:
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds
            best_f1, opt_th = self._get_optimal_f1(y_test, probs)
            roc_auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0
            pr_auc = float(average_precision_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0

            results.append({
                "Model_Source": model_name,
                "Classifier": clf_name,
                "F1_Default_0.5": float(f1_score(y_test, preds, zero_division=0)),
                "Best_F1_Optimal": best_f1,
                "Optimal_Threshold": opt_th,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc
            })

        return {
            "results_df": pd.DataFrame(results),
            "best_f1": max(r["Best_F1_Optimal"] for r in results),
            "best_roc_auc": max(r["ROC_AUC"] for r in results)
        }
