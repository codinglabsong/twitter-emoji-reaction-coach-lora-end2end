"""
Inference script for the TweetEval-Emoji LoRA model.

This module provides a CLI to either evaluate the fine-tuned model on the test split
or predict emojis for given input texts.
"""

import argparse
import logging
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding,
)

from twitter_emoji_reaction_lora.data import load_emoji_dataset, tokenize_and_format
from twitter_emoji_reaction_lora.model import (
    build_base_model,
    build_inference_peft_model,
)
from twitter_emoji_reaction_lora.evaluate import compute_metrics
from twitter_emoji_reaction_lora.utils import set_hf_wandb_environ_vars

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including:
            - model_id: HF Hub repo or local path of the LoRA adapter.
            - mode: 'test' to evaluate on test split; 'predict' to classify texts.
            - batch_size: Batch size for evaluation.
            - texts: List of input strings for prediction mode.
            - top_k: Number of top emojis to return in prediction mode.
    """
    parser = argparse.ArgumentParser(
        description="Run final evaluation or prediction with the TweetEval-Emoji LoRA model"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="roberta-base-with-tweet-eval-emoji",
        help="Hugging Face Hub repo or local path for the LoRA adapter",
    )
    parser.add_argument(
        "--mode",
        choices=["test", "predict"],
        default="test",
        help="`test`: run metrics on test split; `predict`: classify input texts",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        help="One or more input texts for `predict` mode",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to return in `predict` mode",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for running evaluation or prediction.

    - Loads tokenizer and base model.
    - Attaches LoRA adapter.
    - If mode=='test', runs evaluation on the test split and logs metrics.
    - If mode=='predict', runs inference and prints top_k emojis for each input text.
    """
    logging.basicConfig(level=logging.INFO)
    set_hf_wandb_environ_vars()
    cfg = parse_args()

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading tokenizer and base model for '{cfg.model_id}'...")

    # Load tokenizer from adapter repo (includes tokenizer files)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    # Load base RoBERTa and attach the LoRA adapter
    base_model = build_base_model()
    model = build_inference_peft_model(base_model, cfg.model_id)

    if cfg.mode == "test":
        logger.info("Running evaluation on the test set...")
        # Load and tokenize dataset
        ds = load_emoji_dataset()
        ds_tok, _ = tokenize_and_format(ds, cfg.model_id)

        # Set up Trainer for evaluation
        eval_args = TrainingArguments(
            output_dir="./eval_output",
            per_device_eval_batch_size=cfg.batch_size,
            report_to=[],  # disable integrations
            label_names=["labels"],
        )
        data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=ds_tok["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tok,
        )

        metrics = trainer.evaluate()
        logger.info(f"Test metrics: {metrics}")

    else:
        if not cfg.texts:
            raise ValueError("`--texts` must be provided in `predict` mode.")
        logger.info("Running prediction on input texts...")
        pipe = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tok,
            return_all_scores=True,
            function_to_apply="softmax",
            device=device,
        )

        # Map label IDs to emojis
        id2label = {
            0: "❤",
            1: "😍",
            2: "😂",
            3: "💕",
            4: "🔥",
            5: "😊",
            6: "😎",
            7: "✨",
            8: "💙",
            9: "😘",
            10: "📷",
            11: "🇺🇸",
            12: "☀",
            13: "💜",
            14: "😉",
            15: "💯",
            16: "😁",
            17: "🎄",
            18: "📸",
            19: "😜",
        }

        def _emojify(text, k=cfg.top_k):
            """
            Predict top k emojis for the given text.

            Args:
                text (str): Input string.
                k (int): Number of top emojis to return.

            Returns:
                str: Space-separated top k emojis.
            """
            probs = pipe(text)[0]
            top = sorted(probs, key=lambda x: x["score"], reverse=True)[:k]
            return " ".join(id2label[int(d["label"].split("_")[-1])] for d in top)

        for text in cfg.texts:
            logger.info(f"Input: {text}")
            logger.info(f"Output: {_emojify(text)}")


if __name__ == "__main__":
    main()
