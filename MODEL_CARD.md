# Model Card: roberta-base with TweetEval Emoji (LoRA)

## Model Details
- **Base model**: `roberta-base`
- **LoRA rank**: 32
- **Dropout**: 0.05
- **Number of labels**: 20 emoji classes

## Training Data
- [TweetEval emoji dataset](https://huggingface.co/datasets/tweet_eval) containing real tweets.

## Evaluation
Results reported on the test split:

- Accuracy: 0.4252
- F1 macro: 0.3314
- Top‑3 accuracy: 0.6504

(See `README.md` for training commands and more context.)

## Intended Use
This model is provided for experimentation with LoRA adapters and emoji prediction research. It is not a production-ready sentiment or moderation system.

## Limitations
- Trained on public tweets, which may include slang or offensive language.
- English-centric; performance may vary on other languages.
- Predictions may reflect biases present in the training data.

## Citation
If you use this work, please cite the TweetEval dataset and the Hugging Face PEFT documentation.
