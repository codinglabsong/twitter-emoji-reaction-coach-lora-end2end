"""
Train a RoBERTa-based emoji reaction classifier on TweetEval-Emoji,
using PEFT/LoRA adapters and class-weighted loss.  
Exposes configurable hyperparameters and optional push to Hugging Face Hub.
"""

import argparse
import logging
import random
import numpy as np
import torch
import os
import wandb
from datasets import DatasetDict
from huggingface_hub import login
from typing import Callable, Dict
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from twitter_emoji_reaction_lora.data import load_emoji_dataset, tokenize_and_format
from twitter_emoji_reaction_lora.model import build_base_model, build_peft_model
from twitter_emoji_reaction_lora.utils import print_trainable_parameters
from twitter_emoji_reaction_lora.evaluate import compute_metrics
from sklearn.utils.class_weight import compute_class_weight
from uuid import uuid4

logger = logging.getLogger(__name__)

# setting environ vars depending on local or remote training
try:
    # if python-dotenv is installed, load it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # no python-dotenv available (e.g. in SageMaker container), skip
    pass

# log into huggingface
login(token=os.getenv("HUGGINGFACE_TOKEN"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    p = argparse.ArgumentParser()

    # client hyperparams
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    p.add_argument(
        "--peft_rank",
        type=int,
        default=32,
        help="LoRA adapter rank (r) – controls adapter capacity.",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate for optimizer.",
    )
    p.add_argument(
        "--num_train_epochs", type=int, default=4, help="Number of training epochs."
    )
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size per device during training.",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size per device during evaluation.",
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear warm-up.",
    )

    # other client params
    p.add_argument(
        "--model_id",
        type=str,
        default="roberta-base-with-tweet-eval-emoji",
        help="Output directory / model identifier prefix.",
    )
    p.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, push the trained model & tokenizer to the HF Hub.",
    )
    p.add_argument(
        "--hf_hub_repo_id",
        type=str,
        default=None,
        help="Your HF Hub repo (e.g. username/model-name). Required if --push_to_hub.",
    )
    p.add_argument(
        "--skip_test",
        action="store_false",
        dest="do_test",
        help="Skip evaluation on the test split after training.",
    )

    # wandb
    p.add_argument(
        "--wandb-project", type=str, default="Emoji-reaction-coach-with-lora"
    )

    # output directories (special SageMaker paths that rely on Sagemaker's env vars)
    p.add_argument("--model-dir", default=os.getenv("SM_MODEL_DIR", "output/"))

    # validations
    args = p.parse_args()
    if args.push_to_hub and not args.hf_hub_repo_id:
        p.error("--hf_hub_repo_id is required when --push_to_hub is set")

    return args


def get_weighted_trainer(
    model: torch.nn.Module,
    args: TrainingArguments,
    ds_tok: DatasetDict,
    data_collator: DataCollatorWithPadding,
    compute_metrics: Callable[["EvalPrediction"], Dict[str, float]],
    tokenizer: PreTrainedTokenizerBase,
) -> Trainer:
    """
    Return a Trainer with class-balanced cross-entropy loss.

    Args:
        model: a Transformers sequence-classification model.
        args: training arguments.
        ds_tok: a DatasetDict with 'train' and 'validation' splits.
        data_collator: batches examples.
        compute_metrics: fn(EvalPrediction) → {metric_name: value}.
        tokenizer: the tokenizer for saving/pushing.

    Returns:
        A Trainer whose compute_loss applies per-class weights.
    """
    labels = np.array(ds_tok["train"]["labels"])
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
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
    """Seed all RNGs (Python, NumPy, Torch) for deterministic runs."""
    random.seed(seed)  # vanilla Python Random Number Generator (RNG)
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU-side torch RNG
    torch.cuda.manual_seed_all(seed)  # all GPU RNGs
    torch.backends.cudnn.deterministic = True  # force deterministic conv kernels
    torch.backends.cudnn.benchmark = False  # trade speed for reproducibility


def main() -> None:
    """Entry point: parse args, prepare data, train, evaluate, and optionally push."""
    logging.basicConfig(level=logging.INFO)
    cfg = parse_args()

    # ---------- Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {DEVICE}")

    # reproducibility
    set_seed(cfg.seed)
    logger.info(f"Set seed: {cfg.seed}")

    # ---------- Data Preprocessing ----------
    # download and tokenize dataset
    ds = load_emoji_dataset()
    ds_tok, tok = tokenize_and_format(ds)

    # initialize base model and LoRA
    model = build_base_model()
    logger.info(f"Base model trainable params: {print_trainable_parameters(model)}")
    lora_model = build_peft_model(model, cfg.peft_rank)
    logger.info(
        f"LoRA model (peft_rank={cfg.peft_rank}) trainable params: {print_trainable_parameters(lora_model)}"
    )

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
        run_name=f"emoji-{uuid4().hex[:8]}",
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
        logger.info("running final test-set evaluation...")
        metrics = trainer.evaluate(ds_tok["test"])
        logger.info(f"test metrics:\n{metrics}")
    else:
        logger.info("skipping test evaluation.")

    # save model & tokenizer to output_dir
    logger.info("saving LoRA model and tokenizer...")
    trainer.save_model()

    # push to hub
    if cfg.push_to_hub:
        logger.info("pushing to Huggingface hub...")
        trainer.push_to_hub(
            repo_id=cfg.hf_hub_repo_id,
            finetuned_from="FacebookAI/roberta-base",
            tasks="text-classification",
            dataset="tweet_eval/emoji",
        )

    wandb.finish()


if __name__ == "__main__":
    main()
