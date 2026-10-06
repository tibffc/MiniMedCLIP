import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets
import torch
import torch.nn.functional as F

from .config import DATA_ROOT, CACHE_DIR, ROOT, IMG_SIZE, LP_TRAIN_N


CLASSES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]


def collect_image_table():
    rows = []
    for cls in CLASSES:
        for path in sorted((DATA_ROOT / cls / "images").glob("*.png")):
            rows.append({"file": f"{cls}/images/{path.name}", "class_name": cls})
    return pd.DataFrame(rows)


def remove_duplicate_images(all_df):
    all_df = all_df.copy()
    all_df["md5"] = [
        hashlib.md5((DATA_ROOT / rel).read_bytes()).hexdigest()
        for rel in tqdm(all_df["file"], desc="считаю хеши")
    ]
    duplicate_groups = all_df.groupby("md5")["file"].transform("size")
    dups = all_df[duplicate_groups > 1]
    n_groups = dups["md5"].nunique()
    classes_per_group = dups.groupby("md5")["class_name"].nunique()

    info = {
        "n_groups": int(n_groups),
        "n_files": int(len(dups)),
        "cross_class_groups": int((classes_per_group > 1).sum()),
        "class_counts": dups.groupby("md5")["class_name"].first().value_counts(),
    }

    clean = all_df.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    return clean, info


def make_or_load_split(all_df, split_seed=13):
    split_file = ROOT / "splits" / "split_all_classes.csv"
    split_file.parent.mkdir(exist_ok=True)

    if split_file.exists():
        split_df = pd.read_csv(split_file)
    else:
        train_part, temp_part = train_test_split(
            all_df, test_size=0.30, stratify=all_df["class_name"], random_state=split_seed
        )
        val_part, test_part = train_test_split(
            temp_part, test_size=2 / 3, stratify=temp_part["class_name"], random_state=split_seed
        )
        split_df = pd.concat([
            train_part.assign(split="train"),
            val_part.assign(split="val"),
            test_part.assign(split="test"),
        ]).sort_values("file").reset_index(drop=True)
        split_df.to_csv(split_file, index=False)

    return split_df, split_file


def get_binary(df):
    out = df[df["class_name"].isin(["COVID", "Normal"])].copy()
    out["label"] = (out["class_name"] == "COVID").astype(int)
    return out.reset_index(drop=True)


def prepare_radiography_splits(split_seed=13):
    all_df = collect_image_table()
    all_df, duplicate_info = remove_duplicate_images(all_df)
    split_df, split_file = make_or_load_split(all_df, split_seed)
    train_bin = get_binary(split_df[split_df["split"] == "train"])
    val_bin = get_binary(split_df[split_df["split"] == "val"])
    test_bin = get_binary(split_df[split_df["split"] == "test"])
    return {
        "all_df": all_df,
        "split_df": split_df,
        "split_file": split_file,
        "duplicate_info": duplicate_info,
        "train_bin": train_bin,
        "val_bin": val_bin,
        "test_bin": test_bin,
    }


def load_xray_array(df, name):
    path = CACHE_DIR / f"xray64_{name}_{len(df)}.npy"
    if path.exists():
        return np.load(path)
    arr = np.zeros((len(df), 64, 64), dtype=np.uint8)
    for i, rel in enumerate(tqdm(df["file"], desc=f"читаю {name}")):
        with Image.open(DATA_ROOT / rel) as im:
            arr[i] = np.asarray(im.convert("L").resize((64, 64), Image.Resampling.BILINEAR))
    np.save(path, arr)
    return arr


def load_xray_splits(train_bin, val_bin, test_bin):
    xr = {}
    for name, df in [("train", train_bin), ("val", val_bin), ("test", test_bin)]:
        xr[name] = (
            torch.from_numpy(load_xray_array(df, name)),
            torch.from_numpy(df["label"].to_numpy()),
        )
    return xr


def prepare_xray(x_uint8, device):
    x = x_uint8.to(device).float() / 255.0
    x = (x - 0.5) / 0.5
    return x.unsqueeze(1).repeat(1, 3, 1, 1)


def load_cifar_raw(kind, train, root):
    cls = datasets.CIFAR10 if kind == "cifar10" else datasets.CIFAR100
    ds = cls(root=str(root), train=train, download=True)
    return ds.data, np.array(ds.targets), list(ds.classes)


def get_cifar(kind, train, root, n=None, seed=0):
    data, labels, classes = load_cifar_raw(kind, train, root)
    if n is not None and n < len(data):
        idx = np.random.default_rng(seed).permutation(len(data))[:n]
        data, labels = data[idx], labels[idx]
    return torch.from_numpy(data), torch.from_numpy(labels), classes


def prepare_batch(x_uint8, device):
    x = x_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5


def load_cifar_experiment(cifar_dir, n=LP_TRAIN_N):
    c10_train_x, c10_train_y, c10_names = get_cifar("cifar10", True, cifar_dir, n=n, seed=13)
    c10_test_x, c10_test_y, _ = get_cifar("cifar10", False, cifar_dir)
    c100_train_x, c100_train_y, c100_names = get_cifar("cifar100", True, cifar_dir, n=n, seed=13)
    c100_test_x, c100_test_y, _ = get_cifar("cifar100", False, cifar_dir)
    return {
        "c10_train_x": c10_train_x, "c10_train_y": c10_train_y,
        "c10_test_x": c10_test_x, "c10_test_y": c10_test_y,
        "c100_train_x": c100_train_x, "c100_train_y": c100_train_y,
        "c100_test_x": c100_test_x, "c100_test_y": c100_test_y,
        "C10_NAMES": c10_names, "C100_NAMES": c100_names,
    }
