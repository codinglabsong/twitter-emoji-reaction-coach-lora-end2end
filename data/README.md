# notes on where the data lives, how it’s structured

We use the TweetEval Emoji dataset from HuggingFace:

- Split: train / validation / test
- 20 classes (😂, ❤️, etc.)

Run `bash data/download.sh` to fetch and cache it under `./.cache`.
