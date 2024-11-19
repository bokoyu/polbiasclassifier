from sklearn.metrics import classification_report

def compute_metrics(preds, labels, target_names):
    """Generate a classification report."""
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
