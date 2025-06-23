from evaluate import load
import numpy as np

# load once at import time
accuracy_metric = load("accuracy")
f1_metric       = load("f1")

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
    acc_res = accuracy_metric.compute(
        predictions=preds, 
        references=labels
    )

    # 2) Macro-F1
    f1_res  = f1_metric.compute(
        predictions=preds, 
        references=labels, 
        average="macro"
    )

    # 3) Top-3 accuracy
    top3 = np.any(
        np.argsort(logits, axis=-1)[:, -3:] == labels[:, None],
        axis=1
    )
    top3_acc = top3.mean().item()

    # combine results
    return {
        **acc_res,
        **f1_res,
        "top3_accuracy": top3_acc,
    }
