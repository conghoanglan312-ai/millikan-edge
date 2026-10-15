"""Score the detector on apparatus units never used in training -- the unseen-unit test.

    python edge/score_unseen.py               -> results/unseen/SCORE.md, score.json
    python edge/score_unseen.py --todo <unit> -> labels/<unit>/review_todo.json, for
                                                 `edge/label.py review <unit>`

For every unit with role "unseen" in common.FIXTURE_INFO whose hand labels exist:
  truth   labels/<unit>/labels.json from edge/label.py, corrected by review.json once
          the blind second look is done -- score.py's rule for bench_0908
  area    the detection area the pipeline's gates used on that unit (grid lines, panel
          margin, frame edge, static marks), saved by `edge/autolabel.py <unit>
          --labelled`; a label or a prediction inside it is not scored, as on bench_0908
  rows    the classical detector and tracker from that replay, and each model filtering
          the classical blobs at the labelled frames -- the offline filter of
          edge/int8_test.py, within 0.04 false a frame of a full replay on bench_0908
The sample is 25 frames from each quarter of a session, so each quarter weighs the
same, and each unit weighs the same in the pooled row.  95% bootstrap intervals,
resampling frames within each (unit, quarter).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402
from edge.score import BOOT, corrected, pct, prf                    # noqa: E402

OUT = os.path.join(C.RESULTS, "unseen")
SCALE = 32.0                # edge/train.py: the model sees (frame - background) / 32
MODELS = (("FP32 classifier, first model", os.path.join(C.RESULTS, "fp32", "model.onnx")),
          ("FP32 classifier, retrained with four more units",
           os.path.join(C.DATA, "models", "fp32_v2_20261015", "model.onnx")))


class Unit(object):
    """One unseen unit: its labels, its replay, and the area the gates looked at."""

    def __init__(self, name):
        self.name = name
        d = os.path.join(C.REPO, "labels", name)
        self.sample = C.loadJson(os.path.join(d, "sample.json"))
        self.labels = C.loadJson(os.path.join(d, "labels.json"), {"items": {}})
        answers = C.loadJson(os.path.join(d, "review.json"), {"answers": {}})["answers"]
        self.reviewed = bool(answers)
        self.byPos = {}
        for a in answers.values():
            self.byPos.setdefault(a["pos"], []).append(a)
        with np.load(os.path.join(C.DATA, "autolabel", name + ".npz")) as f:
            self.z = {k: f[k] for k in f.files}
        z = self.z
        self.X, self.Y = float(z["roi"][0]), float(z["roi"][1])
        self.W, self.H = [float(v) for v in z["roiWH"]]
        self.rows, self.cols = z["gridRows"], z["gridCols"]
        self.gridClear, self.edgeClear, self.panelClear = [float(v) for v in z["clear"]]
        self.mask = z["staticMask"]
        crop = set(z["cropFrames"].tolist())
        self.items = [it for it in self.sample["items"]
                      if it["repeatOf"] is None and it["frame"] in crop
                      and self.labels["items"].get(str(it["pos"]), {}).get("done")]

    def blanked(self, xy):
        """score.Truth.blanked, on this unit's own detection area."""
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
        """Sure and unsure labels in the replay's ROI pixels (labels are full-frame)."""
        pts = corrected(self.labels["items"].get(str(pos), {}).get("points", []),
                        self.byPos.get(pos, []))
        sure = [[p[0] - self.X, p[1] - self.Y] for p in pts if p[2]]
        unsure = [[p[0] - self.X, p[1] - self.Y] for p in pts if not p[2]]
        return (np.array(sure, float).reshape(-1, 2), np.array(unsure, float).reshape(-1, 2))

    def kept(self):
        return self.z["gate"] == 0

    def detector(self, keep = None):
        k = self.kept()
        f, x, y = self.z["frame"][k], self.z["x"][k], self.z["y"][k]
        if keep is not None:
            f, x, y = f[keep], x[keep], y[keep]
        return byFrame(f, x, y)

    def tracker(self):
        s = self.z["sightings"]
        s = s[s[:, 4] >= int(self.z["confirmPoints"])]
        return byFrame(s[:, 0], s[:, 2], s[:, 3])

    def crops(self):
        k = self.kept()
        d = self.z["gray"][k].astype(np.float32) - self.z["bg"][k].astype(np.float32)
        return np.ascontiguousarray((d / SCALE)[:, 4:28, 4:28][:, None])


def byFrame(f, x, y):
    out = {}
    for fi, xi, yi in zip(f, x, y):
        out.setdefault(int(fi), []).append([float(xi), float(yi)])
    return {k: np.array(v, float) for k, v in out.items()}


def logits(path, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers = ["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.array([sess.run(None, {name: x[k:k + 1]})[0].ravel()[0]
                     for k in range(len(x))])


def perFrame(u, pred):
    """{(unit, quarter): [[tp, fp, fn] per labelled frame]}"""
    per = {}
    for it in u.items:
        p = pred.get(it["frame"], np.zeros((0, 2)))
        s, n = u.truth(it["pos"])
        p, s, n = p[~u.blanked(p)], s[~u.blanked(s)], n[~u.blanked(n)]
        m = C.match(p, s, n)
        per.setdefault((u.name, it["stratum"]), []).append((m["tp"], m["fp"], m["fn"]))
    return {k: np.array(v, float).reshape(-1, 3) for k, v in per.items()}


def weighted(per):
    """Each unit the same weight, each quarter within it the same weight."""
    units = sorted({k[0] for k in per})
    w = {k: 1.0 / len(units) / sum(1 for j in per if j[0] == k[0]) for k in per}
    mean = sum(w[k] * per[k].mean(axis = 0) for k in per)
    rng = np.random.default_rng(20261015)
    acc = np.zeros((BOOT, 3))
    for k, a in per.items():
        acc += w[k] * a[rng.integers(0, len(a), size = (BOOT, len(a)))].mean(axis = 1)
    tp, fp, fn = acc[:, 0], acc[:, 1], acc[:, 2]
    with np.errstate(invalid = "ignore", divide = "ignore"):
        P, R = tp / (tp + fp), tp / (tp + fn)
        F = 2 * P * R / (P + R)
    ci = [tuple(np.nanpercentile(v, [2.5, 97.5])) for v in (P, R, F)]
    return {"counts": mean, "prf": prf(*mean), "ci": ci}


def writeTodo(u, rows):
    """The blind second look's spots: every place a row and the labels disagree --
    the parent's detacc_score.makeReview.  A prediction with no label (false, or a
    droplet the labeller missed) and a label nothing found (missed, or not a droplet)
    look the same to the reviewer, in random order, with no row named."""
    out = []
    for it in u.items:
        s, n = u.truth(it["pos"])
        cand = []
        for pred in rows:
            p = pred.get(it["frame"], np.zeros((0, 2)))
            m = C.match(p, s, n)
            cand += [tuple(p[k]) for k in m["fpIdx"]] + [tuple(s[k]) for k in m["fnIdx"]]
        kept = []
        for x, y in cand:
            fx, fy = x + u.X, y + u.Y
            if any((fx - a) ** 2 + (fy - b) ** 2 <= C.MATCH_PX ** 2 for a, b in kept):
                continue
            if any((fx - a["x"]) ** 2 + (fy - a["y"]) ** 2 <= C.MATCH_PX ** 2
                   for a in u.byPos.get(it["pos"], [])):
                continue
            kept.append((fx, fy))
        out += [{"rid": "%d:%d:%d" % (it["pos"], round(fx), round(fy)), "pos": it["pos"],
                 "frame": it["frame"], "x": round(fx, 2), "y": round(fy, 2)}
                for fx, fy in kept]
    order = np.random.default_rng().permutation(len(out))
    path = os.path.join(C.REPO, "labels", u.name, "review_todo.json")
    C.saveJson(path, {"rows": "classical detector and tracker, every model's detector",
                      "radius": C.MATCH_PX, "items": [out[k] for k in order]})
    print("%d spots to review -> %s" % (len(out), path))


def todo(name):
    u = Unit(name)
    left = [it for it in u.sample["items"]
            if not u.labels["items"].get(str(it["pos"]), {}).get("done")]
    if left:
        sys.exit("%s: %d screens not labelled yet -- finish them first, or the review "
                 "misses their frames" % (name, len(left)))
    rows = [u.detector(), u.tracker()]
    x = u.crops()
    for _name, path in MODELS:
        if os.path.exists(path):
            rows.append(u.detector(logits(path, x) >= 0.0))
    writeTodo(u, rows)


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default = OUT)
    ap.add_argument("--todo", metavar = "UNIT")
    a = ap.parse_args()
    if a.todo:
        todo(a.todo)
        return
    names = [n for n, (_t, role) in C.FIXTURE_INFO.items() if role == "unseen"]
    units = []
    for n in names:
        if not os.path.exists(os.path.join(C.DATA, "autolabel", n + ".npz")):
            print("%s: no replay yet -- python edge/autolabel.py %s --labelled" % (n, n))
            continue
        u = Unit(n)
        print("%s: %d labelled frames%s" % (n, len(u.items),
                                            ", reviewed" if u.reviewed else ""))
        if u.items:
            units.append(u)
    if not units:
        sys.exit("nothing to score yet")

    models = [(name, path) for name, path in MODELS if os.path.exists(path)]
    rows = {"classical detector": {}, "classical tracker": {}}
    rows.update({"%s — detector" % name: {} for name, _p in models})
    for u in units:
        rows["classical detector"][u.name] = perFrame(u, u.detector())
        rows["classical tracker"][u.name] = perFrame(u, u.tracker())
        x = u.crops()
        for name, path in models:
            keep = logits(path, x) >= 0.0                    # p >= 0.5
            rows["%s — detector" % name][u.name] = perFrame(u, u.detector(keep))

    L = ["# Detection accuracy on units never used in training", "",
         "Hand-labelled frames (`edge/label.py`) inside the detection area; each quarter "
         "of a session and each unit weighted the same; 95%% bootstrap intervals (%d "
         "resamples). Match radius %.0f px. Model rows filter the classical blobs at the "
         "labelled frames. Written by `edge/score_unseen.py`." % (BOOT, C.MATCH_PX), ""]
    L += ["| unit | labelled frames | sure droplets in the area | blind review |",
          "|---|---|---|---|"]
    for u in units:
        n = sum(int((~u.blanked(u.truth(it["pos"])[0])).sum()) for it in u.items)
        L.append("| %s | %d | %d | %s |" % (u.name, len(u.items), n,
                                             "done" if u.reviewed else "not yet"))
    L += ["", "| units | row | precision | recall | F1 | false / frame | missed / frame |",
          "|---|---|---|---|---|---|---|"]
    out = {}
    groups = [(u.name, [u.name]) for u in units]
    if len(units) > 1:
        groups.append(("all unseen", [u.name for u in units]))
    for label, members in groups:
        for row, perUnit in rows.items():
            per = {}
            for m in members:
                per.update(perUnit[m])
            w = weighted(per)
            (P, R, F), ci = w["prf"], w["ci"]
            L.append("| %s | %s | %s | %s | %s | %.2f | %.2f |"
                     % (label, row, pct(P, ci[0]), pct(R, ci[1]), pct(F, ci[2]),
                        w["counts"][1], w["counts"][2]))
            out.setdefault(label, {})[row] = {
                "precision": P, "recall": R, "f1": F, "ci": ci,
                "falsePerFrame": float(w["counts"][1]),
                "missedPerFrame": float(w["counts"][2])}
    os.makedirs(a.out, exist_ok = True)
    with open(os.path.join(a.out, "SCORE.md"), "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, "score.json"), "w", encoding = "utf-8") as f:
        json.dump(out, f, indent = 1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
