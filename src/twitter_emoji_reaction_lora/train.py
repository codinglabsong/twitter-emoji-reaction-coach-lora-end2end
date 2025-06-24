import argparse
import random
import numpy as np
import torch
import os
import wandb
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from twitter_emoji_reaction_lora.data import load_emoji_dataset, tokenize_and_format
from twitter_emoji_reaction_lora.model import build_base_model, build_peft_model
from twitter_emoji_reaction_lora.utils import print_trainable_parameters
from twitter_emoji_reaction_lora.evaluate import compute_metrics
from sklearn.utils.class_weight import compute_class_weight
from uuid import uuid4

def parse_args():
    p = argparse.ArgumentParser()

    # client hyperparams
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peft_rank", type=float, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    # other client params
    p.add_argument("--model_id", type=str, default="roberta-base-with-tweet-eval-emoji")
    p.add_argument("--push_to_hub", type=bool, default=False)
    p.add_argument("--hf_hub_repo_id", type=str, default="codinglabsong/roberta-base-tweet-emoji-lora")
    p.add_argument("--do_test", type=bool, default=True)
    
    # wandb
    p.add_argument("--wandb-project", type=str, default="Emoji-reaction-coach-with-lora")

    # output directories (special SageMaker paths that rely on Sagemaker's env vars)
    p.add_argument("--model-dir", default=os.getenv("SM_MODEL_DIR", "output/"))
    return p.parse_args()

def get_weighted_trainer(
    model, 
    args, 
    ds_tok, 
    data_collator, 
    compute_metrics,
    tokenizer
):
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
        tokenizer=tokenizer,
    )

    return trainer


def set_seed(seed: int) -> None:
    """Ensure reproducibility"""
    random.seed(seed)  # vanilla Python Random Number Generator (RNG)
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU-side torch RNG
    torch.cuda.manual_seed_all(seed)  # all GPU RNGs
    torch.backends.cudnn.deterministic = True  # force deterministic conv kernels
    torch.backends.cudnn.benchmark = False  # trade speed for reproducibility


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = parse_args()
    
    # ---------- Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using: {DEVICE}")
    
    # reproducibility
    set_seed(cfg.seed)
    logging.info(f"Set seed: {cfg.seed}")
    
    # environ
    os.environ["WANDB_NOTES"] = "Fine tune model with low rank adaptation for an emoji reaction coach"
    
    # ---------- Data Preprocessing ----------
    # download and tokenize dataset
    ds = load_emoji_dataset()
    ds_tok, tok = tokenize_and_format(ds)
    
    # initialize base model and LoRA
    model = build_base_model()
    logging.info(f"Base model trainable params: {print_trainable_parameters(model)}")
    lora_model = build_peft_model(model, cfg.peft_rank)
    logging.info(f"LoRA model (peft_rank={cfg.peft_rank}) trainable params: {print_trainable_parameters(lora_model)}")
    
    # ---------- Train ----------
    # setup trainer and train
    training_args = TrainingArguments(
        output_dir=cfg.model_id,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        max_grad_norm=0.5,
        label_smoothing_factor=0.1,
        save_total_limit=2,
        logging_steps=30,
        fp16=True,
        push_to_hub=cfg.push_to_hub,
        report_to="wandb",
        run_name=f"copmuter-emoji-{uuid4().hex[:8]}",
        label_names=["labels"],
    )
    data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    
    trainer = get_weighted_trainer(
        model=lora_model,
        args=training_args,
        ds_tok=ds_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tok,
    )

    trainer.train()
    
    # ---------- Save, Test or Push ----------
    # evaluate test 
    if cfg.do_test:
        logging.info("running final test-set evaluation...")
        metrics = trainer.evaluate(ds_tok["test"])
        logging.info(f"Test metrics:\n{metrics}")

    # save model & tokenizer to output_dir
    logging.info("saving LoRA model and tokenizer...")    
    trainer.save_model() 
    
    # push to hub
    if cfg.push_to_hub:
        logging.info("pushing to Huggingface hub...")
        trainer.push_to_hub(
            repo_id=cfg.hf_hub_repo_id,
            finetuned_from="FacebookAI/roberta-base",
            tasks="text-classification",
            dataset="tweet_eval/emoji",
        )
    
    wandb.finish()
    
    
if __name__ == "__main__":
    main()
