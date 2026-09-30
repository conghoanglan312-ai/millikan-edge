"""Replay a raw-frame fixture through the unmodified pipeline and cut labelled crops.

    python edge/autolabel.py panel_sweep3_0909        every 15th frame
    python edge/autolabel.py bench_0908 --labelled    the 150 hand-labelled frames

    -> data/autolabel/<fixture>.npz, data/autolabel/<fixture>.json (summary)

Nothing in cv/ is edited.  Functions are wrapped from outside, as the parent's
test/tools/detacc_replay.py does:
  FramesSource.frames  -> which frame index is being processed
  detectDroplets       -> every contour it found: kept, or rejected and by which gate;
                          on a crop frame, a 32x32 crop of the frame and of the
                          background it was compared against
  Track.__init__/update-> every sighting: frame, track id, position, points so far

A contour's label is decided after the whole fixture has run, from the tracker:
   1  kept, and a sighting of a track that EVER reached CONFIRM_POINTS sightings
   0  kept, but its track never got that far, or it joined none
  -1  rejected by a gate (the gate is in `gate`)
So the labels agree with the tracker by construction.  That is why the hand-labelled
bench_0908 set, which is independent of it, is the only test set.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2                                                           # noqa: E402
from cv import pipeline as P                                         # noqa: E402
from cv import source as S                                           # noqa: E402
from edge import common as C                                         # noqa: E402

HALF = 16                   # 32x32 crops: room for a 24x24 model input plus jitter
EVERY = 15                  # crop frame spacing on pool fixtures, ~0.5 s at 30 fps
MAX_REJECTS = 40            # rejected contours cropped per frame, sampled; the kept
                            # ones are always all cropped
GATES = ("kept", "area<min", "area>max", "circularity", "static mark", "grid row",
         "grid col", "panel margin", "frame edge")

cur = {"i": -1}
cropFrames = set()
rows = []                   # (frame, x, y, area, circ, gate)
grays, bgs = [], []
sightings = []              # (frame, trackId, x, y, pointsSoFar)
meta = {}
rng = np.random.default_rng(20260929)

origFrames = S.FramesSource.frames
origDetect = P.detectDroplets
origInit = P.Track.__init__
origUpdate = P.Track.update
origROI = P.autoROI


def frames(self, first = 0, last = None):
    j = first
    for frame, t in origFrames(self, first, last):
        while j < len(self.times) and float(self.times[j]) != t:
            j += 1
        cur["i"] = j
        yield frame, t
        j += 1


def _cut(padded, x, y):
    ix, iy = int(round(x)) + HALF, int(round(y)) + HALF
    return padded[iy - HALF:iy + HALF, ix - HALF:ix + HALF]


def detectDroplets(gray, background, s, *a, **k):
    out = origDetect(gray, background, s, *a, **k)
    meta.setdefault("roiW", s["roiW"])
    i = cur["i"]
    if i not in cropFrames:
        return out
    rejected = list(P.REJECT_POINTS)
    if len(rejected) > MAX_REJECTS:
        pick = rng.choice(len(rejected), MAX_REJECTS, replace = False)
        rejected = [rejected[k] for k in sorted(pick)]
    g = cv2.copyMakeBorder(gray, HALF, HALF, HALF, HALF, cv2.BORDER_CONSTANT, value = 0)
    b = cv2.copyMakeBorder(np.asarray(background, np.uint8), HALF, HALF, HALF, HALF,
                           cv2.BORDER_CONSTANT, value = 0)
    for cx, cy, area in out[0]:
        rows.append((i, cx, cy, area, np.nan, 0))
        grays.append(_cut(g, cx, cy))
        bgs.append(_cut(b, cx, cy))
    for cx, cy, why, area, circ in rejected:
        rows.append((i, cx, cy, area, circ,
                     GATES.index(why) if why in GATES else len(GATES)))
        grays.append(_cut(g, cx, cy))
        bgs.append(_cut(b, cx, cy))
    return out


def trackInit(self, centre, t):
    origInit(self, centre, t)
    sightings.append((cur["i"], self.id, float(centre[0]), float(centre[1]), 1))


def trackUpdate(self, centre, t, maxSpeed):
    origUpdate(self, centre, t, maxSpeed)
    sightings.append((cur["i"], self.id, float(centre[0]), float(centre[1]),
                      len(self.points)))


def autoROI(fullBackground):
    roi = origROI(fullBackground)
    if roi is not None:
        meta["roi"] = [roi["x"], roi["y"], roi["w"], roi["h"], roi["spacing"]]
    return roi


def croppedROI(folder):
    """one_droplet_slow was saved already cropped, so autoROI finds no grid in it.

    The parent's test/tools/check.py supplies the ROI from roi.txt instead; this is
    that override, copied, so the replay matches the acceptance gate's own run.
    """
    info = dict(l.split() for l in open(os.path.join(folder, "roi.txt")))
    w, h = int(info["w"]), int(info["h"])
    spac, yTop = float(info["spacing"]), int(info["yTop"])
    rows = [int(round(yTop - P.LINE_TOP * spac + k * spac)) for k in range(11)]
    bg = cv2.imread(os.path.join(folder, "background.png"), cv2.IMREAD_GRAYSCALE)
    cm = bg.mean(axis = 0)
    raw = [int(c) for c in np.where(cm > cm.mean() + 2.0 * cm.std())[0]]
    grp = []
    for c in raw:
        if not grp or c - grp[-1][-1] > 3:
            grp.append([c])
        else:
            grp[-1].append(c)
    cols = [int(np.mean(g)) for g in grp]
    meta["roi"] = [0, 0, w, h, spac]
    P.autoROI = lambda *a, **k: {"x": 0, "y": 0, "w": w, "h": h, "spacing": spac,
                                 "rows": rows, "cols": cols, "border": w}
    P.readoutBox = lambda roi: (0, 0, 1, 1)
    P.fieldOffFromCells = lambda cells: None
    P.fieldWasOff = lambda *a, **k: None


def label(rowsArr, sightArr):
    """Each kept row's track and label, matched on the exact centre the tracker got."""
    best = {}
    for tid, n in zip(sightArr[:, 1].astype(int), sightArr[:, 4].astype(int)):
        best[tid] = max(best.get(tid, 0), n)
    where = {(int(f), round(float(x), 3), round(float(y), 3)): int(tid)
             for f, tid, x, y, _n in sightArr}
    track = np.full(len(rowsArr), -1, np.int32)
    lab = np.full(len(rowsArr), -1, np.int8)
    for k, (f, x, y, _a, _c, gate) in enumerate(rowsArr):
        if gate != 0:
            continue
        tid = where.get((int(f), round(float(x), 3), round(float(y), 3)))
        if tid is None:
            lab[k] = 0
            continue
        track[k] = tid
        lab[k] = 1 if best[tid] >= P.CONFIRM_POINTS else 0
    points = np.array([best.get(int(t), 0) for t in track], np.int16)
    return track, lab, points


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fixture")
    ap.add_argument("--labelled", action = "store_true",
                    help = "crop exactly the hand-labelled frames (bench_0908)")
    ap.add_argument("--every", type = int, default = EVERY)
    a = ap.parse_args()

    folder = C.fixture(a.fixture)
    n = len(np.load(os.path.join(folder, "times.npy")))
    if a.labelled:
        sample = C.loadJson(os.path.join(C.LABELS, "sample.json"))
        cropFrames.update(int(it["frame"]) for it in sample["items"])
    else:
        cropFrames.update(range(0, n, a.every))

    S.FramesSource.frames = frames
    P.detectDroplets = detectDroplets
    P.Track.__init__ = trackInit
    P.Track.update = trackUpdate
    P.autoROI = autoROI
    if a.fixture in C.CROPPED:
        croppedROI(folder)
    P._beep = lambda: None
    P.writeLiveStatus = lambda **k: None

    out = os.path.join(C.DATA, "autolabel")
    os.makedirs(out, exist_ok = True)
    t0 = time.perf_counter()
    falls = P.trackAll("frames:" + folder, show = False, saveVideo = False,
                       outDir = os.path.join(out, "runs", a.fixture))
    minutes = (time.perf_counter() - t0) / 60.0

    rowsArr = np.array(rows, np.float64).reshape(-1, 6)
    sightArr = np.array(sightings, np.float64).reshape(-1, 5)
    track, lab, points = label(rowsArr, sightArr)
    fallArr = np.array([(f.id, bool(f.legal), f.seconds or np.nan) for f in falls],
                       np.float64).reshape(-1, 3)
    np.savez_compressed(
        os.path.join(out, a.fixture + ".npz"),
        gray = np.array(grays, np.uint8).reshape(-1, 2 * HALF, 2 * HALF),
        bg = np.array(bgs, np.uint8).reshape(-1, 2 * HALF, 2 * HALF),
        frame = rowsArr[:, 0].astype(np.int32), x = rowsArr[:, 1].astype(np.float32),
        y = rowsArr[:, 2].astype(np.float32), area = rowsArr[:, 3].astype(np.float32),
        circ = rowsArr[:, 4].astype(np.float32), gate = rowsArr[:, 5].astype(np.int8),
        label = lab, track = track, trackPoints = points,
        gates = np.array(GATES), sightings = sightArr.astype(np.float32),
        falls = fallArr, cropFrames = np.array(sorted(cropFrames), np.int32),
        roi = np.array(meta.get("roi", [np.nan] * 5), np.float64),
        confirmPoints = np.int32(P.CONFIRM_POINTS))
    summary = {
        "fixture": a.fixture, "captured": C.FIXTURE_INFO.get(a.fixture, ("?", "?"))[0],
        "role": C.FIXTURE_INFO.get(a.fixture, ("?", "?"))[1],
        "frames": n, "cropFrames": len(cropFrames), "minutes": round(minutes, 1),
        "rows": len(rowsArr), "positive": int((lab == 1).sum()),
        "keptUnconfirmed": int((lab == 0).sum()), "rejected": int((lab == -1).sum()),
        "byGate": {g: int((rowsArr[:, 5] == k).sum()) for k, g in enumerate(GATES)},
        "tracks": int(len(set(sightArr[:, 1].astype(int)))) if len(sightArr) else 0,
        "falls": len(falls), "legalFalls": int(sum(1 for f in falls if f.legal)),
        # what check.py compares on a cropped fixture, which has no panel to be legal
        "kinematicLegalSeconds": sorted(round(f.seconds, 2) for f in falls
                                        if f.kinematicLegal),
    }
    with open(os.path.join(out, a.fixture + ".json"), "w", encoding = "utf-8") as f:
        json.dump(summary, f, indent = 1)
    print(json.dumps(summary, indent = 1), flush = True)


if __name__ == "__main__":
    main()
