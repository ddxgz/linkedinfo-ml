import pandas as pd
from sklearn import metrics


def classification_report_avg(y_true, y_pred):
    report = metrics.classification_report(
        y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    cols_avg = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
    return df_report.loc[cols_avg]
