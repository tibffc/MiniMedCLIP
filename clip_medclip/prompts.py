import torch
import pandas as pd

from .text import SimpleTextTokenizer


PROMPT_SETS = {
    "A_radiology": [
        ("COVID-19 pneumonia with bilateral peripheral ground-glass opacities.", 1),
        ("Chest radiograph demonstrates findings compatible with COVID-19 infection.", 1),
        ("Bilateral pulmonary opacities in a patient with COVID-19 pneumonia.", 1),
        ("Peripheral lower-zone airspace opacity consistent with COVID-19.", 1),
        ("No acute cardiopulmonary abnormality. No focal pulmonary opacity.", 0),
        ("Normal chest radiograph without acute findings.", 0),
        ("No evidence of pneumonia or focal airspace disease.", 0),
        ("Lungs are clear and there is no acute abnormality.", 0),
    ],
}

ZERO_SHOT_PROMPTS = {
    1: ["Chest radiograph with COVID-19 pneumonia.", "COVID-19 infection.", "Findings of COVID-19 pneumonia."],
    0: ["Normal chest radiograph.", "No acute abnormality.", "Lungs are clear."],
}

ZERO_SHOT_BY_SET = {"A_radiology": ZERO_SHOT_PROMPTS}
_prompt_cache = {}


def get_prompt_data(set_name):
    if set_name not in _prompt_cache:
        pool = PROMPT_SETS[set_name]
        tok = SimpleTextTokenizer([text for text, _ in pool])
        ids = torch.stack([tok.encode(text) for text, _ in pool])
        y = torch.tensor([label for _, label in pool])
        sem = torch.stack([y, 1 - y], dim=1).float()
        _prompt_cache[set_name] = (tok, ids, sem)
    return _prompt_cache[set_name]


def prompt_table(set_name):
    return pd.DataFrame(PROMPT_SETS[set_name], columns=["текст", "метка (1=COVID)"])
