"""
Fidelity Metric Plugin: Kolmogorov-Smirnov (KS) test and Wasserstein Distance.
Evaluates 1D feature marginal distribution similarity against ground-truth real anomalies.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from ..core.registry import register_metric
from ..core.base_metric import BaseMetric


@register_metric("fidelity")
class FidelityMetric(BaseMetric):
    """
    Computes Kolmogorov-Smirnov (KS) statistic and Wasserstein distance
    between synthetic samples and real test anomalies.
    Lower is better (0 = identical distributions).
    """
    def evaluate(
        self,
        real_norm: np.ndarray,
        real_anom: np.ndarray,
        synth_anom: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        num_features = min(real_anom.shape[1], synth_anom.shape[1])
        ks_scores = []
        w_scores = []

        for i in range(num_features):
            r_feat = real_anom[:, i]
            s_feat = synth_anom[:, i]
            ks_stat = ks_2samp(r_feat, s_feat).statistic
            w_dist = wasserstein_distance(r_feat, s_feat)
            ks_scores.append(ks_stat)
            w_scores.append(w_dist)

        return {
            "mean_ks": float(np.mean(ks_scores)),
            "mean_wasserstein": float(np.mean(w_scores)),
            "per_feature_ks": {name: float(ks) for name, ks in zip(feature_names[:num_features], ks_scores)},
            "per_feature_wasserstein": {name: float(w) for name, w in zip(feature_names[:num_features], w_scores)}
        }
