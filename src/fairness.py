### src/fairness.py
import numpy as np
import pandas as pd

def evaluate_fairness(preds, sensitive_features):
    # Binarize sensitive feature: 1 if alcohol >= 10 else 0
    bins = [0, 10, float('inf')]
    group = pd.cut(sensitive_features, bins=bins, labels=[0, 1]).astype(int)

    group_0_preds = np.array(preds)[group == 0]
    group_1_preds = np.array(preds)[group == 1]

    avg_0 = np.mean(group_0_preds)
    avg_1 = np.mean(group_1_preds)

    fairness_score = abs(avg_0 - avg_1)
    return {"statistical_parity_difference": fairness_score}

