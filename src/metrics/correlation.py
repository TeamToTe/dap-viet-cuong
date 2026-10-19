"""
Correlation Preservation Metric Plugin.
Measures inter-sensor correlation matrix preservation via Frobenius norm difference
for both Pearson and Spearman rank correlation.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.stats import spearmanr

from ..core.registry import register_metric
from ..core.base_metric import BaseMetric


@register_metric("correlation")
class CorrelationMetric(BaseMetric):
    """
    Computes Frobenius norm differences:
    || R_real - R_synth ||_F (Pearson)
    || S_real - S_synth ||_F (Spearman)
    Lower indicates better preservation of multi-sensor physical couplings.
    """
    def evaluate(
        self,
        real_norm: np.ndarray,
        real_anom: np.ndarray,
        synth_anom: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        # Pearson correlation
        r_p = np.nan_to_num(np.corrcoef(real_anom, rowvar=False))
        s_p = np.nan_to_num(np.corrcoef(synth_anom, rowvar=False))
        delta_pearson_fro = float(np.linalg.norm(r_p - s_p, ord='fro'))

        # Spearman rank correlation
        r_s, _ = spearmanr(real_anom)
        s_s, _ = spearmanr(synth_anom)
        r_s = np.nan_to_num(r_s)
        s_s = np.nan_to_num(s_s)
        delta_spearman_fro = float(np.linalg.norm(r_s - s_s, ord='fro'))

        return {
            "delta_pearson_frobenius": delta_pearson_fro,
            "delta_spearman_frobenius": delta_spearman_fro,
            "real_pearson_corr": r_p,
            "synth_pearson_corr": s_p
        }
