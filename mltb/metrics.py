import pandas as pd
from sklearn import metrics
from typing import List


def classification_report_avg(y_true, y_pred, cols_avg: List[str] = None):
    report = metrics.classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    if cols_avg is not None:
        cols = cols_avg
    else:
        cols = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']

    return df_report.loc[cols, ]
