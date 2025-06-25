"""
Module with helper functions for inspecting and logging properties of HuggingFace Transformer models.
"""

import os
import logging
from huggingface_hub import login
from transformers import PreTrainedModel

# setting environ vars depending on local or remote training
try:
    # if python-dotenv is installed, load it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # no python-dotenv available (e.g. in SageMaker container), skip
    pass

logger = logging.getLogger(__name__)


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: any HF Transformers model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"


def set_hf_wandb_environ_vars(wandb_project: str = "Emoji-reaction-coach-with-lora"):
    """
    Configure authentication for Hugging Face Hub and set the Weights & Biases project.

    1. If the HUGGINGFACE_TOKEN environment variable is present (or loaded from .env),
       log into the Hugging Face Hub using that token.
    2. In all cases, set the WANDB_PROJECT environment variable so W&B runs are grouped
       under the given project name.

    Args:
        wandb_project: Name of the W&B project to use for logging.
    """
    # hf
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged into Hugging Face Hub")
    else:
        logger.info("No HF token found to log into Hugging Face Hub")

    # wandb
    os.environ["WANDB_PROJECT"] = wandb_project
