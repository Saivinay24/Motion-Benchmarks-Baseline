import numpy as np
from scipy.stats import pearsonr, spearmanr

class AlignmentEngine:
    """
    Project 2306 - Human-Alignment Statistics.
    Validates VLM scores against Ground Truth Human Preference data.
    """
    @staticmethod
    def calculate_alignment(vlm_scores, human_scores):
        """
        Performs statistical correlation analysis.
        """
        # Pearson: Linear correlation
        pearson_r, p_p = pearsonr(vlm_scores, human_scores)
        
        # Spearman: Rank-order correlation (Crucial for preference-based tasks)
        spearman_rho, p_s = spearmanr(vlm_scores, human_scores)
        
        results = {
            "pearson_r": pearson_r,
            "pearson_p": p_p,
            "spearman_rho": spearman_rho,
            "spearman_p": p_s,
            "is_significant": p_s < 0.05
        }
        
        print(f"Alignment Report:\nSpearman Rho: {spearman_rho:.4f} (p={p_s:.4f})")
        if results['is_significant']:
            print("VLM demonstrates statistically significant alignment with human judgment.")
        
        return results