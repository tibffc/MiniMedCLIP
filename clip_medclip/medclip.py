import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import CKPT_DIR, EMBED_DIM, MEDCLIP_EPOCHS
from .utils import DEVICE, seed_everything
from .data import prepare_xray
from .miniclip import MiniCLIP
from .prompts import get_prompt_data, ZERO_SHOT_BY_SET, ZERO_SHOT_PROMPTS


def augment(x):
    B, dev = x.size(0), x.device
    angle = (torch.rand(B, device=dev) - 0.5) * 2 * math.radians(8)
    scale = 0.9 + 0.15 * torch.rand(B, device=dev)
    tx = (torch.rand(B, device=dev) - 0.5) * 0.1
    ty = (torch.rand(B, device=dev) - 0.5) * 0.1
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], dim=1),
        torch.stack([sin * flip, cos, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    if dev.type == "mps":
        h, w = x.shape[-2:]
        limits = grid.new_tensor([1 - 1 / w, 1 - 1 / h])
        grid = torch.maximum(torch.minimum(grid, limits), -limits)
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    else:
        x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    contrast = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=dev)
    brightness = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    return (x * contrast + brightness).clamp(-1, 1)


def semantic_targets(a_labels, b_labels, mode="normalized"):
    a = F.normalize(a_labels.float(), dim=-1)
    b = F.normalize(b_labels.float(), dim=-1)
    sim = a @ b.T
    if mode == "softmax":
        return F.softmax(sim, dim=1)
    return sim / sim.sum(dim=1, keepdim=True).clamp_min(1e-8)


def semantic_matching_loss(img_emb, txt_emb, img_labels, txt_labels, logit_scale, mode="normalized"):
    logits = logit_scale.exp().clamp(max=100) * img_emb @ txt_emb.T
    target_i2t = semantic_targets(img_labels, txt_labels, mode)
    target_t2i = semantic_targets(txt_labels, img_labels, mode)
    loss_i2t = -(target_i2t * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_t2i = -(target_t2i * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return (loss_i2t + loss_t2i) / 2


def xray_features(model, x_uint8, batch=1000):
    with torch.no_grad():
        model.eval()
        parts = [
            model.encode_image(prepare_xray(x_uint8[i:i + batch], DEVICE)).cpu()
            for i in range(0, len(x_uint8), batch)
        ]
        return torch.cat(parts).numpy()


def xray_zero_shot(model, tokenizer, feats, labels, prompts=ZERO_SHOT_PROMPTS):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    with torch.no_grad():
        model.eval()
        class_vecs = []
        for cls in [0, 1]:
            ids = torch.stack([tokenizer.encode(p) for p in prompts[cls]]).to(DEVICE)
            class_vecs.append(F.normalize(model.encode_text(ids).mean(dim=0), dim=0))
        text_feats = torch.stack(class_vecs).cpu().numpy()

    sims = feats @ text_feats.T
    pred = sims.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
    }
    return metrics, sims


def train_medclip(seed, xr_train_x, xr_val_x, xr_val_y, prompt_set="A_radiology",
                  target_mode="normalized", epochs=None, batch_size=64,
                  lr=3e-4, weight_decay=1e-4, verbose=True):
    epochs = epochs or MEDCLIP_EPOCHS
    tok, pool_ids, pool_sem = get_prompt_data(prompt_set)
    ckpt_path = CKPT_DIR / "medclip" / f"{prompt_set}__{target_mode}" / f"seed_{seed}.pt"

    seed_everything(seed)
    model = MiniCLIP(len(tok.stoi), EMBED_DIM).to(DEVICE)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        return model.eval(), state["history"]

    pool_ids, pool_sem = pool_ids.to(DEVICE), pool_sem.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    gen = torch.Generator().manual_seed(seed)
    n, history = len(xr_train_x), []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        order = torch.randperm(n, generator=gen)
        losses = []
        for i in range(0, n - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            images = augment(prepare_xray(xr_train_x[idx], DEVICE))
            y = xr_train_y[idx].to(DEVICE)
            img_labels = torch.stack([y, 1 - y], dim=1).float()
            t_idx = torch.randint(0, len(pool_ids), (batch_size,), generator=gen).to(DEVICE)

            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(pool_ids[t_idx])
            loss = semantic_matching_loss(
                img_emb, txt_emb, img_labels, pool_sem[t_idx],
                model.logit_scale, target_mode
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        metrics, _ = xray_zero_shot(
            model, tok, xray_features(model, xr_val_x),
            xr_val_y.numpy(), ZERO_SHOT_BY_SET[prompt_set]
        )
        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_accuracy": metrics["accuracy"],
        })
        if verbose:
            print(
                f"seed={seed} эпоха {epoch + 1:2d}/{epochs}  "
                f"loss={np.mean(losses):.4f}  "
                f"val balanced acc={metrics['balanced_accuracy']:.3f}  "
                f"({time.time() - t0:.0f} c)"
            )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "history": history, "seed": seed,
        "prompt_set": prompt_set, "target_mode": target_mode, "epochs": epochs,
    }, ckpt_path)
    return model.eval(), history
