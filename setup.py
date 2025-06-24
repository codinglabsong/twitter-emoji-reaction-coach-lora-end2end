from setuptools import setup, find_packages

setup(
    name="twitter_emoji_reaction_lora",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers",
        "datasets",
        "peft",
        "evaluate",
        "python-dotenv",
        "numpy",
        "wandb",
        "scikit-learn",
        "ipykernel",
    ],
)
