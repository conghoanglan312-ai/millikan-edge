"""Is this track a droplet?  A person answers, one key per clip.

    python edge/review_tracks.py prepare     sample tracks, cut clips  (~2 min, once)
    python edge/review_tracks.py review      y droplet / n not / u unsure / b back / q quit
    python edge/review_tracks.py report      the answers, per recording and track length

Why: the tracker's "confirmed" is the auto-label, and how often it is a real droplet
differs by recording (HANDOVER §4, measured against the hand labels on bench_0908
and a noise proxy elsewhere).  A person looking at the track itself settles it.

What is shown: the track's own sightings, up to 16 spread over its life, a 96x96
window centred on it, enlarged 3x, looping.  A droplet is a small bright dot that
stays one dot, frame after frame, while the speckle around it changes; a phantom is
speckle that the tracker happened to string together.  Blind: the recording, the
track's length and its label are not shown, and clips come in random order.

Sampled: from the recordings with droplets, confirmed tracks in three length
bands (6-49, 50-199, 200+ sightings).  The droplet-free sweeps need no review: of
their 42,190 tracks none lasted more than 3 sightings (2026-09-30), so there is
nothing a person could follow in them.  bench_0908 is never shown: it is the test
set.  Answers resume where you stopped.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402

OUT = os.path.join(C.DATA, "review")
CLIPS = os.path.join(OUT, "clips.npz")
ANSWERS = os.path.join(OUT, "answers.json")
BANDS = ((6, 50), (50, 200), (200, 10 ** 9))
PER_BAND = 20
FRAMES = 16
HALF = 48
SCALE = 3
STRETCH = np.clip(np.arange(256) * 2.5 - 30, 0, 255).astype(np.uint8)
DROPLETS = ("one_droplet_slow", "zoom_labelled_0907", "spray_labelled_0907",
            "unit2-4_1009")


def bandName(n):
    for lo, hi in BANDS:
        if lo <= n < hi:
            return "%d-%s" % (lo, hi - 1 if hi < 10 ** 9 else "+")
    return "<6"


def tracksOf(name):
    with np.load(os.path.join(C.DATA, "autolabel", name + ".npz")) as z:
        s, roi = z["sightings"], z["roi"]
    ids = s[:, 1].astype(np.int64)
    order = np.argsort(ids, kind = "stable")
    s, ids = s[order], ids[order]
    uniq, first = np.unique(ids, return_index = True)
    last = list(first[1:]) + [len(ids)]
    return {int(u): s[a:b] for u, a, b in zip(uniq, first, last)}, roi


def clip(name, rows, roi):
    folder = C.fixture(name)
    pick = np.unique(np.linspace(0, len(rows) - 1, min(FRAMES, len(rows))).astype(int))
    ox, oy = (0, 0) if name in C.CROPPED else (int(roi[0]), int(roi[1]))
    out = []
    for k in pick:
        f, _tid, x, y, _n = rows[k]
        g = cv2.imread(os.path.join(folder, "f%06d.png" % int(f)), cv2.IMREAD_GRAYSCALE)
        g = cv2.copyMakeBorder(g, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value = 0)
        cx, cy = int(round(x)) + ox + HALF, int(round(y)) + oy + HALF
        out.append(cv2.LUT(g[cy - HALF:cy + HALF, cx - HALF:cx + HALF], STRETCH))
    while len(out) < FRAMES:
        out.append(out[-1])
    return np.array(out, np.uint8)


def prepare():
    rng = np.random.default_rng(20260930)
    clips, meta = [], []
    for name in DROPLETS:
        tracks, roi = tracksOf(name)
        length = {t: int(r[:, 4].max()) for t, r in tracks.items()}
        chosen = []
        for lo, hi in BANDS:
            ids = [t for t, n in length.items() if lo <= n < hi]
            chosen += list(rng.choice(ids, min(PER_BAND, len(ids)), replace = False))
        for t in chosen:
            clips.append(clip(name, tracks[t], roi))
            meta.append({"fixture": name, "track": int(t), "points": length[t],
                         "band": bandName(length[t])})
        print("%-20s %d clips" % (name, len(chosen)), flush = True)
    order = rng.permutation(len(clips))
    os.makedirs(OUT, exist_ok = True)
    np.savez_compressed(CLIPS, clips = np.array(clips)[order],
                        meta = np.array([json.dumps(meta[k]) for k in order]))
    print("-> %s  (%d clips)" % (CLIPS, len(clips)))


def load():
    z = np.load(CLIPS)
    return z["clips"], [json.loads(m) for m in z["meta"]]


def review():
    clips, meta = load()
    answers = C.loadJson(ANSWERS, {})
    k = next((i for i in range(len(clips)) if str(i) not in answers), len(clips))
    win = "droplet?  y yes / n no / u unsure / b back / q quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    t = 0
    while k < len(clips):
        f = cv2.resize(clips[k][t % FRAMES], None, fx = SCALE, fy = SCALE,
                       interpolation = cv2.INTER_NEAREST)
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        c = HALF * SCALE
        for a, b in (((c - 30, c), (c - 12, c)), ((c + 12, c), (c + 30, c)),
                     ((c, c - 30), (c, c - 12)), ((c, c + 12), (c, c + 30))):
            cv2.line(f, a, b, (0, 170, 255), 1)
        bar = np.zeros((28, f.shape[1], 3), np.uint8)
        cv2.putText(bar, "%d / %d" % (k + 1, len(clips)), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, np.vstack([bar, f]))
        key = cv2.waitKey(120) & 0xFF
        t += 1
        if key in (ord("y"), ord("n"), ord("u")):
            answers[str(k)] = {"a": chr(key), **meta[k]}
            with open(ANSWERS, "w", encoding = "utf-8") as fh:
                json.dump(answers, fh, indent = 1)
            k, t = k + 1, 0
        elif key == ord("b") and k > 0:
            k, t = k - 1, 0
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    print("%d of %d answered -> %s" % (len(answers), len(clips), ANSWERS))


def report():
    answers = C.loadJson(ANSWERS, {})
    groups = {}
    for a in answers.values():
        groups.setdefault((a["fixture"], a["band"]), []).append(a["a"])
    L = ["| recording | tracks | shown | droplet | not | unsure |", "|---|---|---|---|---|---|"]
    for (name, band), v in sorted(groups.items()):
        L.append("| %s | %s | %d | %d | %d | %d |" % (name, band, len(v), v.count("y"),
                                                     v.count("n"), v.count("u")))
    text = "\n".join(L)
    print(text)
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding = "utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    {"prepare": prepare, "review": review, "report": report}[sys.argv[1]]()
