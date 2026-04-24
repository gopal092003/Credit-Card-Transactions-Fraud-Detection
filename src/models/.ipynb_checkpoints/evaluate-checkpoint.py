from sklearn.metrics import f1_score, recall_score, classification_report


def evaluate_model(y_true, y_pred):
    """
    Return evaluation metrics
    """
    return {
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }