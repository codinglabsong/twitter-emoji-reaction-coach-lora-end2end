import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel

# Specify the same model ID pushed or trained locally
MODEL_ID = "roberta-base-with-tweet-eval-emoji"

# Load tokenizer + model
device = 0 if torch.cuda.is_available() else -1
tok = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-base",
    num_labels=20,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(base_model, MODEL_ID).eval()

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


def predict_emojis(text, top_k=3):
    """
    Predict top k emojis for the given text.

    Args:
        text (str): Input string.
        k (int): Number of top emojis to return.

    Returns:
        str: Space-separated top k emojis.
    """
    probs = pipe(text)[0]
    top = sorted(probs, key=lambda x: x["score"], reverse=True)[:top_k]
    return " ".join(id2label[int(d["label"].split("_")[-1])] for d in top)


# 4) Build the Gradio interface
demo = gr.Interface(
    fn=predict_emojis,
    inputs=[
        gr.Textbox(label="Enter a X/Twitter post here"),
        gr.Slider(minimum=1, maximum=5, step=1, label="Top-K Emojis"),
    ],
    outputs=gr.Textbox(label="Reply with Emojis..."),
    title="TweetEval Emoji Reaction Coach",
    description="Type any tweet-like text and get the top-k emoji reactions!",
    examples=[["Sunny days!"], ["That movie was amazing."]],
)

if __name__ == "__main__":
    demo.launch()
