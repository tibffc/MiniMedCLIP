import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

ROOT = Path.cwd()
DATA_ROOT = ROOT / "data" / "COVID-19_Radiography_Dataset"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CKPT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "cache"

for folder in [RESULTS_DIR, FIG_DIR, CKPT_DIR, CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Не нашёл данные в {DATA_ROOT}")

SEEDS = [13, 42, 2024, 43, 44, 45, 46, 2025, 2026, 2027]

IMG_SIZE = 64
EMBED_DIM = 64
LP_TRAIN_N = 10_000
MEDCLIP_EPOCHS = 20
