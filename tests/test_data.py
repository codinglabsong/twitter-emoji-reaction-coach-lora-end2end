from datasets import Dataset, DatasetDict
import twitter_emoji_reaction_lora.data as data


def test_load_emoji_dataset(monkeypatch):
    def fake_load_dataset(name, subset):
        dummy = Dataset.from_dict({"text": ["hi"], "label": [0]})
        return DatasetDict(train=dummy, validation=dummy, test=dummy)

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    ds = data.load_emoji_dataset()
    assert list(ds.keys()) == ["train", "validation", "test"]