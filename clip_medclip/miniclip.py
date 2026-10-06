import math
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CKPT_DIR, EMBED_DIM, ROOT
from .text import SimpleTextTokenizer, CIFAR10_PROMPTS, make_prompt
from .utils import DEVICE, seed_everything
from .data import prepare_batch


class SmallTextEncoder(nn.Module):
    def __init__(self, vocab_size, width=64, layers=1, heads=4):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, 32, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, ids):
        x = self.token(ids) + self.pos[:, :ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=ids.eq(0))
        mask = ids.ne(0).unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.norm(x)


class SmallImageEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, width), nn.LayerNorm(width),
        )

    def forward(self, x):
        return self.net(x)


class MiniCLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.image_encoder = SmallImageEncoder(embed_dim)
        self.text_encoder = SmallTextEncoder(vocab_size, embed_dim)
        self.image_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        return F.normalize(self.image_projection(self.image_encoder(images)), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.text_projection(self.text_encoder(ids)), dim=-1)

    def forward(self, images, ids):
        return self.encode_image(images), self.encode_text(ids)


def clip_loss(image_emb, text_emb, logit_scale):
    logits = logit_scale.exp().clamp(max=100) * image_emb @ text_emb.T
    targets = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def build_model(tokenizer):
    return MiniCLIP(len(tokenizer.stoi), EMBED_DIM).to(DEVICE)


def _checkpoint_dirs():
    zip_path = ROOT / "checkpoints_last.zip"
    old_dir = CKPT_DIR / "checkpoints_last" / "miniclip_cifar10"
    new_dir = CKPT_DIR / "miniclip_cifar10"
    if not old_dir.exists() and zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(CKPT_DIR)
    return new_dir, old_dir


def find_checkpoint(seed):
    new_dir, old_dir = _checkpoint_dirs()
    for folder in [new_dir, old_dir]:
        path = folder / f"seed_{seed}.pt"
        if path.exists():
            return path
    return None


def train_miniclip(seed, train_x, train_y, tokenizer, epochs=20, batch_size=64,
                   lr=3e-4, weight_decay=1e-4):
    seed_everything(seed)
    model = build_model(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_text_ids = torch.stack([tokenizer.encode(p) for p in CIFAR10_PROMPTS]).to(DEVICE)

    generator = torch.Generator().manual_seed(seed)
    n = len(train_x)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        losses = []
        for i in range(0, n - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            images = prepare_batch(train_x[idx], DEVICE)
            flip = torch.rand(len(idx), device=DEVICE) < 0.5
            images = torch.where(flip[:, None, None, None], images.flip(-1), images)
            labels = train_y[idx].to(DEVICE)
            image_emb, text_emb = model(images, class_text_ids[labels])
            loss = clip_loss(image_emb, text_emb, model.logit_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print(f"seed={seed} эпоха {epoch + 1}/{epochs}  loss={np.mean(losses):.4f}")

    new_dir, _ = _checkpoint_dirs()
    new_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epochs - 1, "seed": seed},
               new_dir / f"seed_{seed}.pt")
    return model


def load_miniclip(seed, train_x, train_y, tokenizer):
    path = find_checkpoint(seed)
    if path is None:
        print(f"Чекпоинта для seed={seed} нет, обучаю с нуля...")
        return train_miniclip(seed, train_x, train_y, tokenizer).eval()
    model = build_model(tokenizer)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval()


def make_untrained(seed, tokenizer):
    seed_everything(seed)
    return build_model(tokenizer).eval()


def image_features(model, x_uint8, batch=1000):
    with torch.no_grad():
        model.eval()
        parts = [
            model.encode_image(prepare_batch(x_uint8[i:i + batch], DEVICE)).cpu()
            for i in range(0, len(x_uint8), batch)
        ]
        return torch.cat(parts).numpy()


def zero_shot(model, img_feats, labels, class_names, tokenizer):
    with torch.no_grad():
        model.eval()
        ids = torch.stack([tokenizer.encode(make_prompt(n)) for n in class_names]).to(DEVICE)
        text_feats = model.encode_text(ids).cpu().numpy()
    sims = img_feats @ text_feats.T
    labels = np.asarray(labels)
    top1 = float((sims.argmax(axis=1) == labels).mean())
    top5_idx = np.argsort(-sims, axis=1)[:, :5]
    top5 = float((top5_idx == labels[:, None]).any(axis=1).mean())
    return top1, top5, sims


def fit_and_score(train_x, train_y, test_x, test_y, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    clf.fit(train_x, train_y)
    return float((clf.predict(test_x) == test_y).mean())


def sample_k_per_class(labels, k, rng):
    idx = []
    for c in np.unique(labels):
        class_idx = np.flatnonzero(labels == c)
        idx.extend(rng.choice(class_idx, size=min(k, len(class_idx)), replace=False))
    return np.array(idx)
