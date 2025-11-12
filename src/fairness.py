import numpy as np
import pandas as pd

def evaluate_fairness(preds, sensitive_features):
    # I like to binarize the sensitive feature so I can compare two groups
    # I'm using "10" as a cutoff for the 'alcohol' feature
    bins = [0, 10, float('inf')]
    group = pd.cut(sensitive_features, bins=bins, labels=[0, 1]).astype(int)

    # Here I'm splitting the predictions into two groups based on the bin
    group_0_preds = np.array(preds)[group == 0]
    group_1_preds = np.array(preds)[group == 1]

    # Then I calculate the average predictions for each group
    avg_0 = np.mean(group_0_preds)
    avg_1 = np.mean(group_1_preds)

    # The fairness score is the absolute difference in these averages
    fairness_score = abs(avg_0 - avg_1)

    # I'm returning a dictionary so it's easy to add more fairness metrics later
    return {"statistical_parity_difference": fairness_score}
