"""
Data loading and preprocessing for the TweetEval-Emoji task.
Provides `load_emoji_dataset` and `tokenize_and_format`.
"""

from datasets import DatasetDict, load_dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Tuple


def load_emoji_dataset(cache_dir: str = ".cache") -> DatasetDict:
    """Load the TweetEval Emoji classification dataset.

    Downloads and prepares the “emoji” subset of the TweetEval benchmark
    from Hugging Face Datasets, returning a DatasetDict with splits:

    Returns:
        DatasetDict: A dict-like object with keys "train", "validation", and "test".
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    return load_dataset(
        "tweet_eval",
        "emoji",
        cache_dir=str(cache_path),
    )


def tokenize_and_format(
    ds: DatasetDict,
    checkpoint: str = "FacebookAI/roberta-base",
    max_length: int = 128,
) -> Tuple[DatasetDict, AutoTokenizer]:
    """
    Tokenize and prepare a text dataset for PyTorch training.

    Loads a pretrained tokenizer from `checkpoint`, applies it to the "text"
    field of each example (with truncation and padding to `max_length`), renames
    the "label" column to "labels", and configures the dataset to return
    PyTorch tensors for `input_ids`, `attention_mask`, and `labels`.

    Args:
        ds (DatasetDict): Raw dataset containing "text" and "label" columns.
        checkpoint (str): HuggingFace tokenizer checkpoint to load.
        max_length (int): Maximum token length for padding/truncation.

    Returns:
        Tuple[DatasetDict, AutoTokenizer]:
            - ds_tok: Tokenized and formatted dataset ready for Trainer.
            - tok:   The loaded AutoTokenizer instance.
    """
    tok = AutoTokenizer.from_pretrained(checkpoint)

    def _tokenize(batch):
        return tok(
            batch["text"], truncation=True, padding="max_length", max_length=max_length
        )

    ds_tok = ds.map(_tokenize, batched=True)
    ds_tok = ds_tok.rename_column("label", "labels")
    # making sure that downstream Trainer sees torch tensors
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds_tok, tok
