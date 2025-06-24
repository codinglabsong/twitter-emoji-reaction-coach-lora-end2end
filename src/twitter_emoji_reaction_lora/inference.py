import argparse
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding,
)
from peft import PeftModel

from twitter_emoji_reaction_lora.data import load_emoji_dataset, tokenize_and_format
from twitter_emoji_reaction_lora.model import build_base_model, build_inference_peft_model
from twitter_emoji_reaction_lora.evaluate import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run final evaluation or prediction with the TweetEval-Emoji LoRA model")
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


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = parse_args()
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"Loading tokenizer and base model for '{cfg.model_id}'...")

    # Load tokenizer from adapter repo (includes tokenizer files)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    # Load base RoBERTa and attach the LoRA adapter
    base_model = build_base_model()
    model = build_inference_peft_model(base_model, cfg.model_id)

    if cfg.mode == "test":
        logging.info("Running evaluation on the test set...")
        # Load and tokenize dataset
        ds = load_emoji_dataset()
        ds_tok, _ = tokenize_and_format(ds)

        # Set up Trainer for evaluation
        eval_args = TrainingArguments(
            output_dir="./eval_output",
            per_device_eval_batch_size=cfg.batch_size,
            report_to=[],  # disable integrations
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
        logging.info(f"Test metrics: {metrics}")

    else:
        if not cfg.texts:
            raise ValueError("`--texts` must be provided in `predict` mode.")
        logging.info("Running prediction on input texts...")
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
            0: "❤", 1: "😍", 2: "😂", 3: "💕", 4: "🔥",
            5: "😊", 6: "😎", 7: "✨", 8: "💙", 9: "😘",
            10: "📷", 11: "🇺🇸", 12: "☀", 13: "💜", 14: "😉",
            15: "💯", 16: "😁", 17: "🎄", 18: "📸", 19: "😜"
        }
        
        def _emojify(text, k=cfg.top_k):
            probs = pipe(text)[0]
            top = sorted(probs, key=lambda x: x["score"], reverse=True)[:k]
            return " ".join(id2label[int(d["label"].split("_")[-1])] for d in top)
        
        for text in cfg.texts:
            logging.info(f"Input: {text}")
            logging.info(f"Output: {_emojify(text)}")


if __name__ == "__main__":
    main()
