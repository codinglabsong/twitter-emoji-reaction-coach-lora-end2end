from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    """
    Compute accuracy, macro-F1 (over 20 emoji classes), and top-3 accuracy.
    
    Args:
        eval_pred: Tuple of (logits, labels) from Trainer.predict/evaluate.
    Returns:
        Dict[str, float] with keys "accuracy", "f1", and "top3_accuracy".
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # 1) Accuracy
    acc = accuracy_score(labels, preds)

    # 2) Macro-F1
    f1 = f1_score(labels, preds, average="macro")

    # 3) Top-3 accuracy
    top3 = np.any(
        np.argsort(logits, axis=-1)[:, -3:] == labels[:, None],
        axis=1
    )
    top3_acc = top3.mean().item()

    # combine results
    return {
        "accuracy": acc,
        "f1": f1,
        "top3_accuracy": top3_acc,
    }
