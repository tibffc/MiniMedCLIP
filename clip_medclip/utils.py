import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_fig(name: str) -> None:
    from .config import FIG_DIR
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
