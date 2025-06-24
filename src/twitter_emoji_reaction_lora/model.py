from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel


def build_base_model(
    checkpoint: str = "FacebookAI/roberta-base",
    num_labels: int = 20,
) -> AutoModelForSequenceClassification:
    """
    Load a pre-trained RoBERTa sequence-classification model.

    Args:
        checkpoint: HF model ID
        num_labels: number of classes for the classification head
    Returns:
        A RobertaForSequenceClassification instance
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # suppress head-init warning
    )
    return model


def build_peft_model(
    base_model: AutoModelForSequenceClassification,
    r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    target_modules: list[str] = ("query", "key", "value"),
    modules_to_save: list[str] = ("classifier",),
) -> PeftModel:
    """
    Wrap a base model in a LoRA adapter.

    Args:
        model: a seq-classification model with frozen weights
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: dropout for LoRA layers
        bias: whether to train bias ("none", "all", "lora_only")
        target_modules: which submodules to inject LoRA into
        modules_to_save: modules whose original weights remain trainable
    Returns:
        A PeftModel instance with LoRA adapters
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=list(modules_to_save),
    )

    lora_model = get_peft_model(base_model, config)
    return lora_model


def build_inference_peft_model(
    base_model: AutoModelForSequenceClassification,
    model_id: str = "roberta-base-with-tweet-eval-emoji",
):
    inference_model = PeftModel.from_pretrained(base_model, model_id).eval()

    return inference_model
