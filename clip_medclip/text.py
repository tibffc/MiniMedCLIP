import re
import torch


CIFAR10_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def make_prompt(class_name):
    return f"a photo of a {class_name.replace('_', ' ')}"


CIFAR10_PROMPTS = [make_prompt(n) for n in CIFAR10_NAMES]

VOCAB_EXTRA_TEXTS = [
    "a normal chest X-ray", "a normal chest radiograph", "a chest X-ray with no disease",
    "a COVID-19 chest X-ray", "a chest radiograph showing COVID-19",
    "a chest X-ray with COVID-19 pneumonia",
]


class SimpleTextTokenizer:
    def __init__(self, texts, max_len=32):
        words = set()
        for text in texts:
            words.update(re.findall(r"[a-z0-9-]+", text.lower()))
        self.pad, self.unk = 0, 1
        self.stoi = {"<pad>": self.pad, "<unk>": self.unk}
        for word in sorted(words):
            self.stoi[word] = len(self.stoi)
        self.max_len = max_len

    def tokenize(self, text):
        return re.findall(r"[a-z0-9-]+", text.lower())

    def encode(self, text):
        ids = [self.stoi.get(w, self.unk) for w in self.tokenize(text)][: self.max_len]
        ids += [self.pad] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
