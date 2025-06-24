# shell script to grab the TweetEval data

#!/usr/bin/env bash
set -e

# 1. install all deps
pip install -r requirements.txt

# 2. use HF datasets to download and cache locally
python - <<'PYCODE'
from twitter_emoji_reaction_lora.data import load_emoji_dataset
ds = load_emoji_dataset()
print(f"Loaded splits: {list(ds.keys())}")
PYCODE
echo "Raw data stored in ./.cache"