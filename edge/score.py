"""Score droplet predictions against the hand labels of bench_0908 -- the fork's gate.

    python edge/score.py --freeze     once: write results/baseline/eval_bench_0908.npz
    python edge/score.py              score everything available, write the report

This is the parent's test/tools/detacc_score.py reduced to what a detector change
needs: the same labels, the same blind review, the same matching rule and radius,
the same detection area, the same phase weighting and bootstrap.  Nothing about how
a prediction is judged may differ between the classical row and a model's row, so
the region and the labels are frozen into one file and every row is scored from it.

Rows scored:
  classical, 2026-09-11 code   the parent's replay (test/output/detacc/replay.npz);
                               must reproduce HANDOVER §8.5 exactly, which is what
                               proves this port is faithful
  classical, current code      data/autolabel/bench_0908.npz from edge/autolabel.py,
                               i.e. the cv/pipeline.py the classifier will be put into
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402

EVAL = os.path.join(C.RESULTS, "baseline", "eval_bench_0908.npz")
PARENT_REPLAY = os.path.join(C.PARENT, "test", "output", "detacc", "replay.npz")
BOOT = 2000


def labelledItems():
    sample = C.loadJson(os.path.join(C.LABELS, "sample.json"))
    labels = C.loadJson(os.path.join(C.LABELS, "labels.json"))
    return sample, labels


def freeze():
    """The evaluation region and the 09-11 predictions, cut down to labelled frames."""
    if os.path.exists(EVAL):
        sys.exit("%s exists; it is frozen on purpose. Delete it by hand to redo." % EVAL)
    z = np.load(PARENT_REPLAY)
    sample, _labels = labelledItems()
    frames = np.array(sorted({int(it["frame"]) for it in sample["items"]}), np.int64)

    def keep(a):
        return a[np.isin(a[:, 0].astype(np.int64), frames)]
    os.makedirs(os.path.dirname(EVAL), exist_ok = True)
    np.savez_compressed(
        EVAL, roi = z["roi"], gridRows = z["gridRows"], gridCols = z["gridCols"],
        clear = z["clear"], staticMask = z["staticMask"], times = z["times"],
        seen = z["seen"][np.isin(z["seen"], frames)], frames = frames,
        det = keep(z["det"]), trk = keep(z["trk"]), ring = keep(z["ring"]))
    print("-> %s" % EVAL)


class Truth(object):
    """Labels, blind review and detection area, exactly as detacc_score.Data has them."""

    def __init__(self):
        z = np.load(EVAL)
        self.X, self.Y, self.W, self.H = [float(v) for v in z["roi"][:4]]
        self.rows, self.cols = z["gridRows"], z["gridCols"]
        self.gridClear, self.edgeClear, self.panelClear = [float(v) for v in z["clear"]]
        self.mask = z["staticMask"]
        names = np.array([C.stratum(float(t)) for t in z["times"]])
        self.share = {s: float(np.mean(names == s)) for s in C.STRATA}
        self.z = z
        self.sample, self.labels = labelledItems()
        self.answers = C.loadJson(os.path.join(C.LABELS, "review.json"))["answers"]
        seen = set(z["seen"].tolist())
        done = [it for it in self.sample["items"]
                if self.labels["items"].get(str(it["pos"]), {}).get("done")]
        self.items = [it for it in done if it["frame"] in seen and it["repeatOf"] is None]
        self.byPos = {}
        for a in self.answers.values():
            self.byPos.setdefault(a["pos"], []).append(a)

    def blanked(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.zeros(len(xy), bool)
        if not len(xy):
            return out
        x, y = xy[:, 0], xy[:, 1]
        if len(self.rows):
            out |= np.min(np.abs(y[:, None] - self.rows[None, :]), axis = 1) <= self.gridClear
        if len(self.cols):
            out |= np.min(np.abs(x[:, None] - self.cols[None, :]), axis = 1) <= self.gridClear
        out |= x >= self.W - self.panelClear
        out |= np.minimum(np.minimum(x, y), np.minimum(self.W - x, self.H - y)) <= self.edgeClear
        ix = np.clip(np.round(x).astype(int), 0, self.mask.shape[1] - 1)
        iy = np.clip(np.round(y).astype(int), 0, self.mask.shape[0] - 1)
        out |= self.mask[iy, ix] > 0
        return out

    def truth(self, pos):
        """Sure and unsure labels in ROI pixels, corrected by the blind review."""
        pts = [list(p) for p in self.labels["items"].get(str(pos), {}).get("points", [])]
        for a in self.byPos.get(pos, []):
            d = [(p[0] - a["x"]) ** 2 + (p[1] - a["y"]) ** 2 for p in pts]
            k = int(np.argmin(d)) if d and min(d) <= C.MATCH_PX ** 2 else None
            if a["a"] == "y":
                if k is None:
                    pts.append([a["x"], a["y"], 1])
                else:
                    pts[k][2] = 1
            elif a["a"] == "n":
                if k is not None:
                    pts.pop(k)
            elif k is None:
                pts.append([a["x"], a["y"], 0])
            else:
                pts[k][2] = 0
        sure = [[p[0] - self.X, p[1] - self.Y] for p in pts if p[2]]
        unsure = [[p[0] - self.X, p[1] - self.Y] for p in pts if not p[2]]
        return (np.array(sure, float).reshape(-1, 2), np.array(unsure, float).reshape(-1, 2))


def byFrame(arr, cols = (1, 2)):
    out = {}
    for r in arr:
        out.setdefault(int(r[0]), []).append([float(r[cols[0]]), float(r[cols[1]])])
    return {k: np.array(v, float) for k, v in out.items()}


def perFrame(T, pred, radius = C.MATCH_PX):
    per = {s: [] for s in C.STRATA}
    for it in T.items:
        p = pred.get(it["frame"], np.zeros((0, 2)))
        s, u = T.truth(it["pos"])
        p, s, u = p[~T.blanked(p)], s[~T.blanked(s)], u[~T.blanked(u)]
        m = C.match(p, s, u, radius)
        per[it["stratum"]].append((m["tp"], m["fp"], m["fn"]))
    return {k: np.array(v, float).reshape(-1, 3) for k, v in per.items()}


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp > 0 else float("nan")
    R = tp / (tp + fn) if tp + fn > 0 else float("nan")
    F = 2 * P * R / (P + R) if P == P and R == R and P + R > 0 else float("nan")
    return P, R, F


def weighted(T, per):
    live = [s for s in C.STRATA if len(per[s])]
    w = np.array([T.share[s] for s in live])
    w = w / w.sum()
    mean = sum(wi * per[s].mean(axis = 0) for wi, s in zip(w, live))
    rng = np.random.default_rng(20260911)            # the parent's seed
    acc = np.zeros((BOOT, 3))
    for wi, s in zip(w, live):
        a = per[s]
        acc += wi * a[rng.integers(0, len(a), size = (BOOT, len(a)))].mean(axis = 1)
    tp, fp, fn = acc[:, 0], acc[:, 1], acc[:, 2]
    with np.errstate(invalid = "ignore", divide = "ignore"):
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F = 2 * P * R / (P + R)
    ci = [tuple(np.nanpercentile(v, [2.5, 97.5])) for v in (P, R, F)]
    return {"counts": mean, "prf": prf(*mean), "ci": ci}


def pct(v, c = None):
    s = "–" if v != v else "%.1f%%" % (100 * v)
    return s if c is None else "%s (%.1f–%.1f)" % (s, 100 * c[0], 100 * c[1])


def current(T):
    """The current code's detector and tracker stages, from edge/autolabel.py."""
    path = os.path.join(C.DATA, "autolabel", "bench_0908.npz")
    if not os.path.exists(path):
        return []
    z = np.load(path)
    # Its ROI may differ from the frozen one by a pixel: move into the frozen frame.
    dx, dy = float(z["roi"][0]) - T.X, float(z["roi"][1]) - T.Y
    kept = z["gate"] == 0
    det = np.stack([z["frame"][kept], z["x"][kept] + dx, z["y"][kept] + dy], axis = 1)
    s = z["sightings"]
    s = s[s[:, 4] >= int(z["confirmPoints"])]
    trk = np.stack([s[:, 0], s[:, 2] + dx, s[:, 3] + dy], axis = 1)
    return [("classical detector, current code", byFrame(det)),
            ("classical tracker, current code", byFrame(trk))]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action = "store_true")
    ap.add_argument("--out", default = os.path.join(C.RESULTS, "baseline",
                                                    "SCORE_bench_0908.md"))
    a = ap.parse_args()
    if a.freeze:
        freeze()
        return
    T = Truth()
    z = T.z
    rows = [("classical detector, 2026-09-11 code", byFrame(z["det"])),
            ("classical tracker, 2026-09-11 code", byFrame(z["trk"][z["trk"][:, 4] == 0])),
            ("rings shown to the operator, 2026-09-11 code",
             byFrame(z["ring"][z["ring"][:, 3] == 9]))]
    rows += current(T)
    L = ["# Detection accuracy on bench_0908 — the fork's gate", "",
         "%d hand-labelled frames, after the blind review, inside the detection area; "
         "phases weighted by share of session time; 95%% bootstrap intervals (%d "
         "resamples). Match radius %.0f px. Written by `edge/score.py`." %
         (len(T.items), BOOT, C.MATCH_PX), "",
         "| row | precision | recall | F1 | false / frame | missed / frame |",
         "|---|---|---|---|---|---|"]
    for name, pred in rows:
        w = weighted(T, perFrame(T, pred))
        (P, R, F), ci = w["prf"], w["ci"]
        L.append("| %s | %s | %s | %s | %.2f | %.2f |"
                 % (name, pct(P, ci[0]), pct(R, ci[1]), pct(F, ci[2]),
                    w["counts"][1], w["counts"][2]))
    L += ["", "Inputs (sha256, first 16 hex):", ""]
    for f in ("labels.json", "review.json", "sample.json"):
        L.append("- `cv/for_report/detection_accuracy/%s` %s"
                 % (f, sha(os.path.join(C.LABELS, f))))
    L.append("- `results/baseline/eval_bench_0908.npz` %s" % sha(EVAL))
    text = "\n".join(L) + "\n"
    with open(a.out, "w", encoding = "utf-8") as f:
        f.write(text)
    print(text)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
