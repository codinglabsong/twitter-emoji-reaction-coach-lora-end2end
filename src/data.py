from datasets import load_dataset

def load_emoji_dataset():
    """Load TweetEval-Emoji and return train/validation/test splits."""
    return load_dataset("tweet_eval", "emoji")
