from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

def load_emoji_dataset():
    """Load TweetEval-Emoji and return train/validation/test splits."""
    return load_dataset("tweet_eval", "emoji")


def tokenize_and_format(
    ds: DatasetDict,
    checkpoint: str = "FacebookAI/roberta-base",
    max_length: int = 128,
) -> (DatasetDict, AutoTokenizer):
    """
    Tokenizes, pads/truncates, renames label column, and sets PyTorch format.
    Returns the tokenized DatasetDict and the tokenizer.
    """
    tok = AutoTokenizer.from_pretrained(checkpoint)

    def _tokenize(batch):
        return tok(batch["text"], 
                   truncation=True,
                   padding="max_length",
                   max_length=max_length)

    ds_tok = ds.map(_tokenize, batched=True)
    ds_tok = ds_tok.rename_column("label", "labels")
    # making sure that downstream Trainer sees torch tensors
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds_tok, tok