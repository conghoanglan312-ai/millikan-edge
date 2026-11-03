"""Label-free first look at MJPG vs raw: replay one recording through the pipeline
with the FP32 classifier after the gates (as on the board) and log, per frame, what
the detector found and what the classifier kept, plus each blob's peak contrast.

    python edge/mjpg_check.py <fixture> <outdir> [last frame]
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from cv import pipeline as P                                         # noqa: E402
from edge import common as C                                         # noqa: E402
from edge import fast as F                                           # noqa: E402
from edge import gate as G                                           # noqa: E402

name, out = sys.argv[1], sys.argv[2]
last = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok = True)

gate = G.Gate(os.path.join(REPO, "results", "fp32", "model.onnx"))
orig = P.detectDroplets
rows = []          # per frame: blobs before the classifier, kept after
blobs = []         # per blob: probability, peak contrast (frame - background, 5x5 max)


def detectDroplets(gray, background, s, *a, **k):
    res = orig(gray, background, s, *a, **k)
    centres = res[0]
    p = gate.probs(gray, background, centres) if len(centres) else []
    bg = np.asarray(background, np.int16)
    for c, q in zip(centres, p):
        x, y = int(round(c[0])), int(round(c[1]))
        win = gray[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].astype(np.int16) \
            - bg[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
        blobs.append((float(q), int(win.max()) if win.size else 0))
    kept = [c for c, q in zip(centres, p) if q >= gate.threshold]
    rows.append((len(centres), len(kept)))
    return (kept,) + tuple(res[1:])


P.detectDroplets = detectDroplets
F.install(P)
P._beep = lambda: None
P.writeLiveStatus = lambda **k: None
P.publishFrame = lambda *a, **k: None

t0 = time.time()
P.trackAll("frames:" + C.fixture(name), show = False, saveVideo = False,
           outDir = os.path.join(out, "session"), toFrame = last)
r = np.array(rows, float)
b = np.array(blobs, float) if blobs else np.zeros((0, 2))
drop = b[b[:, 0] >= 0.5] if len(b) else b
noise = b[b[:, 0] < 0.5] if len(b) else b
summary = {
    "fixture": name, "frames": len(rows), "wall_s": round(time.time() - t0, 1),
    "blobs_per_frame": round(float(r[:, 0].mean()), 2),
    "kept_per_frame": round(float(r[:, 1].mean()), 2),
    "kept_share": round(float(r[:, 1].sum() / max(r[:, 0].sum(), 1)), 3),
    "droplet_like": int(len(drop)),
    "droplet_contrast_p25_p50_p75": [float(v) for v in np.percentile(drop[:, 1], [25, 50, 75])] if len(drop) else None,
    "noise_contrast_p50": float(np.median(noise[:, 1])) if len(noise) else None,
    "prob_hist_0_to_1_by_0.1": np.histogram(b[:, 0], bins = 10, range = (0, 1))[0].tolist() if len(b) else None,
}
with open(os.path.join(out, "summary.json"), "w") as f:
    json.dump(summary, f, indent = 1)
np.save(os.path.join(out, "blobs.npy"), b)
np.save(os.path.join(out, "rows.npy"), r)
print(json.dumps(summary, indent = 1), flush = True)
