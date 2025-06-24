from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

def get_weighted_trainer(model, args, ds_tok, data_collator, compute_metrics):
    """
    Create a Hugging Face Trainer that applies class‐balanced weights to the loss.

    This function computes per‐class weights based on the training split of a
    tokenized `DatasetDict`, then defines and returns a custom Trainer subclass
    (`WeightedTrainer`) that uses those weights in its cross‐entropy loss.

    Args:
        model (PreTrainedModel): A Transformers model instantiated for
            sequence classification.
        args (TrainingArguments): The training configuration arguments.
        ds_tok (DatasetDict): A dataset dictionary with keys
            "train", "validation", etc., where ds_tok["train"]["labels"]
            contains the class labels.
        data_collator (DataCollator): A data collator for batching examples.
        compute_metrics (callable): A function that takes an `EvalPrediction`
            and returns a dict of metric names to values.

    Returns:
        Trainer: An instance of a `WeightedTrainer` with:
            - class‐balanced cross‐entropy loss, and
            - the provided model, args, datasets, collator, and metrics.
    """
    labels   = np.array(ds_tok["train"]["labels"])
    weights  = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss
        
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer
