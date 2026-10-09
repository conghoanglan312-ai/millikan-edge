"""Train the droplet-vs-noise patch classifier (FP32) and export it to ONNX.

Runs in WSL (native ARM64 torch 2.2, the RKNN toolchain's venv):

    ~/venvs/rknn/bin/python /mnt/c/Users/LAN/millikan-edge/edge/train.py
    ... --smoke          one short epoch on a subset, to prove the plumbing

    -> data/models/<run>/  model.pt, model.onnx, history.csv, metrics.json, val_scores.npz

Input: the 24x24 centre of a crop, frame minus background, scaled.  That is what the
classical detector thresholds, so the model sees the same evidence the gates did.
Training jitters the window by up to 3 px and flips it; validation uses the centre.

Sampling: each epoch draws droplet and noise equally, and within each class every
(recording, kind) group equally -- otherwise unit2-4_1009, 94% of the droplets,
would decide what a droplet is, and the 2,807 phantom tracks would drown among
gate rejects.

Model selection uses VALIDATION only (09-07, 09-08 evening): the epoch with the
best average precision is kept.  The test set (bench_0908 hand labels) is not
touched here; it is scored once, through edge/score.py, after the classifier is in
the pipeline.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SIDE = 24
JITTER = 3
SCALE = 32.0            # diff / SCALE: a droplet peaks around 20-60 grey levels


def model():
    """~15k parameters, plain conv/BN/ReLU/pool: every op the RKNN toolkit supports."""
    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding = 1), nn.BatchNorm2d(o), nn.ReLU()]
    return nn.Sequential(*block(1, 8), *block(8, 16), nn.MaxPool2d(2),
                         *block(16, 32), nn.MaxPool2d(2), *block(32, 32),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1))


def load(split):
    z = np.load(os.path.join(DATA, "dataset", split + ".npz"))
    diff = (z["gray"].astype(np.float32) - z["bg"].astype(np.float32)) / SCALE
    return {"x": diff, "y": z["y"].astype(np.float32), "kind": z["kind"],
            "rec": z["rec"], "recordings": list(z["recordings"]), "kinds": list(z["kinds"])}


def crop(x, off):
    return x[:, off:off + SIDE, off:off + SIDE]


def augment(x, rng):
    n = len(x)
    out = np.empty((n, SIDE, SIDE), np.float32)
    ox = rng.integers(4 - JITTER, 4 + JITTER + 1, n)
    oy = rng.integers(4 - JITTER, 4 + JITTER + 1, n)
    for k in range(n):
        out[k] = x[k, oy[k]:oy[k] + SIDE, ox[k]:ox[k] + SIDE]
    flipX, flipY = rng.random(n) < 0.5, rng.random(n) < 0.5
    out[flipX] = out[flipX, :, ::-1]
    out[flipY] = out[flipY, ::-1, :]
    return out * rng.uniform(0.8, 1.25, (n, 1, 1)).astype(np.float32)


def weights(d):
    """Equal mass per class, and per (recording, kind) group within a class."""
    w = np.zeros(len(d["y"]))
    for y in (0, 1):
        groups = {}
        for i in np.where(d["y"] == y)[0]:
            groups.setdefault((int(d["rec"][i]), int(d["kind"][i])), []).append(i)
        for rows in groups.values():
            w[rows] = 0.5 / len(groups) / len(rows)
    return w


def ap_auc(score, y):
    order = np.argsort(-score)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    ap = float((precision * y).sum() / max(1, y.sum()))
    pos, neg = y.sum(), len(y) - y.sum()
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    auc = float((ranks[order][y == 1].sum() - pos * (pos + 1) / 2) / max(1, pos * neg))
    return ap, auc


def evaluate(net, d):
    net.eval()
    x = torch.from_numpy(np.ascontiguousarray(crop(d["x"], 4)))[:, None]
    with torch.no_grad():
        s = torch.cat([torch.sigmoid(net(x[i:i + 4096])).flatten()
                       for i in range(0, len(x), 4096)]).numpy()
    y = d["y"]
    ap, auc = ap_auc(s, y)
    p = s >= 0.5
    tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
    out = {"ap": ap, "auc": auc, "precision@0.5": tp / max(1, tp + fp),
           "recall@0.5": tp / max(1, tp + fn)}
    # how each kind fares at 0.5: droplets should pass, every kind of noise should not
    for k, name in enumerate(d["kinds"]):
        m = d["kind"] == k
        if m.any():
            out["pass rate, %s" % name] = float(p[m].mean())
    return out, s


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type = int, default = 40)
    ap.add_argument("--per-epoch", type = int, default = 60000)
    ap.add_argument("--batch", type = int, default = 256)
    ap.add_argument("--lr", type = float, default = 2e-3)
    ap.add_argument("--smoke", action = "store_true")
    ap.add_argument("--out", default = None)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.per_epoch = 1, 4096
    torch.manual_seed(20261009)
    torch.set_num_threads(os.cpu_count())
    rng = np.random.default_rng(20261009)
    run = a.out or os.path.join(DATA, "models", ("smoke" if a.smoke else
                                                 time.strftime("fp32_%Y%m%d_%H%M")))
    os.makedirs(run, exist_ok = True)

    train, val = load("train"), load("val")
    w = weights(train)
    net = model()
    print("parameters: %d" % sum(p.numel() for p in net.parameters()), flush = True)
    opt = torch.optim.AdamW(net.parameters(), lr = a.lr, weight_decay = 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = a.epochs)
    lossf = nn.BCEWithLogitsLoss()
    best, history = None, []
    for epoch in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        net.train()
        idx = rng.choice(len(w), a.per_epoch, p = w / w.sum())
        total = 0.0
        for i in range(0, len(idx), a.batch):
            b = idx[i:i + a.batch]
            x = torch.from_numpy(augment(train["x"][b], rng))[:, None]
            y = torch.from_numpy(train["y"][b])
            opt.zero_grad()
            loss = lossf(net(x).flatten(), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        sched.step()
        m, scores = evaluate(net, val)
        row = {"epoch": epoch, "loss": total / len(idx), "seconds": time.perf_counter() - t0, **m}
        history.append(row)
        print("epoch %2d  loss %.4f  val AP %.4f  AUC %.4f  P %.3f  R %.3f  (%.0f s)"
              % (epoch, row["loss"], m["ap"], m["auc"], m["precision@0.5"],
                 m["recall@0.5"], row["seconds"]), flush = True)
        if best is None or m["ap"] > best["ap"]:
            best = {"epoch": epoch, **m}
            torch.save(net.state_dict(), os.path.join(run, "model.pt"))
            np.savez_compressed(os.path.join(run, "val_scores.npz"), score = scores,
                                y = val["y"], kind = val["kind"], rec = val["rec"])

    with open(os.path.join(run, "history.csv"), "w", newline = "") as f:
        wr = csv.DictWriter(f, fieldnames = list(history[0]))
        wr.writeheader()
        wr.writerows(history)
    net.load_state_dict(torch.load(os.path.join(run, "model.pt")))
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 1, SIDE, SIDE), os.path.join(run, "model.onnx"),
                      input_names = ["patch"], output_names = ["logit"], opset_version = 12)
    meta = {"best": best, "epochs": a.epochs, "perEpoch": a.per_epoch, "batch": a.batch,
            "lr": a.lr, "side": SIDE, "scale": SCALE, "jitter": JITTER,
            "torch": torch.__version__, "train": int(len(train["y"])),
            "val": int(len(val["y"])), "recordings": train["recordings"]}
    with open(os.path.join(run, "metrics.json"), "w", encoding = "utf-8") as f:
        json.dump(meta, f, indent = 1)
    print("best epoch %d: val AP %.4f  -> %s" % (best["epoch"], best["ap"], run), flush = True)


if __name__ == "__main__":
    main()
