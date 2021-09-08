
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics

from collections import namedtuple

ALGORITHM = "Algorithm"
AUC_ROC = "AUC ROC"
AVG_PREC = "Avg. Precision"
TIME = "Time"

Results = namedtuple("Results", "predictions fpr tpr thresholds")

def plot_curve(test_set, label2preds, label2times=None, output_basepath=None):
    label2times = label2times or dict()
    test_labels = np.array([x['is_active'] for x in test_set ], dtype=np.bool)
    
    scores_df = pd.DataFrame()
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 8))
    
    to_save = dict()
    for label, predictions in label2preds.items():
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_labels, predictions)
        to_save[label] = Results(predictions, fpr, tpr, thresholds)
        
        plt.plot(fpr, tpr, label=label)
        roc_auc_score = sklearn.metrics.roc_auc_score(test_labels, predictions)
        avg_prec = sklearn.metrics.average_precision_score(test_labels, predictions)
        scores_df = scores_df.append(pd.DataFrame(
                   [[label,           roc_auc_score, avg_prec, label2times.get(label, np.nan)]],
            columns=[ALGORITHM,       AUC_ROC,       AVG_PREC, TIME]
        ))
            
        
    plt.plot([0, 1], [0, 1], '--', c='black')
    plt.legend()
    if output_basepath is not None:
        plt.savefig(output_basepath + ".pdf")
        plt.savefig(output_basepath + ".png")
        pd.to_pickle([test_labels, to_save], output_basepath + ".pickle")
    plt.close()
    
    # for label, predictions in label2preds.items():
        # precision, recall, _ = sklearn.metrics.precision_recall_curve(test_labels, np.exp(predictions))
        # plt.plot(recall, precision, label=label)

    return scores_df, output_basepath
