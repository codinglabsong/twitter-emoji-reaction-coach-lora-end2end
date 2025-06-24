from transformers import AutoModelForSequenceClassification, RobertaConfig
from peft.peft_model import PeftModel
import twitter_emoji_reaction_lora.model as model


def test_build_base_and_peft(monkeypatch):
    def fake_from_pretrained(checkpoint, num_labels, ignore_mismatched_sizes=True):
        config = RobertaConfig(num_labels=num_labels)
        return AutoModelForSequenceClassification.from_config(config)

    monkeypatch.setattr(
        AutoModelForSequenceClassification, "from_pretrained", fake_from_pretrained
    )

    base = model.build_base_model()
    assert base.config.num_labels == 20

    peft = model.build_peft_model(base, r=2, lora_alpha=2)
    assert isinstance(peft, PeftModel)
