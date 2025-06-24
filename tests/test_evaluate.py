import numpy as np
from twitter_emoji_reaction_lora.evaluate import compute_metrics


def test_compute_metrics_perfect():
    logits = np.array([[0.1, 0.9], [0.9, 0.1]])
    labels = np.array([1, 0])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
