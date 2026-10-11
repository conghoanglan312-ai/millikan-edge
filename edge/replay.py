"""Replay bench_0908 through the pipeline, with or without the classifier gate,
and record what the detector and the tracker held at the hand-labelled frames.

    python edge/replay.py classical
    python edge/replay.py after          classifier judges what the gates kept
    python edge/replay.py replace        classifier instead of area + circularity

    -> data/replays/<variant>.npz, read by edge/score.py

Same hooks as edge/autolabel.py, from outside cv/: the frame index, every blob
detectDroplets() returns (after the gate, if one is installed), and every track
sighting with the track's points so far -- the tracker stage counts a sighting once
its track has CONFIRM_POINTS of them, as the parent's replay did.  Only rows at the
labelled frames are kept.  The whole session is replayed: the rolling background
and the tracker need every frame.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv import pipeline as P                                         # noqa: E402
from cv import source as S                                           # noqa: E402
from edge import common as C                                         # noqa: E402
from edge import gate as G                                           # noqa: E402

FIXTURE = "bench_0908"
MODEL = os.path.join(C.RESULTS, "fp32", "model.onnx")

cur = {"i": -1}
want = set()
det, sightings, meta = [], [], {}
origFrames = S.FramesSource.frames
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


def recordOn(detect):
    def detectDroplets(gray, background, s, *a, **k):
        out = detect(gray, background, s, *a, **k)
        if cur["i"] in want:
            for cx, cy, area in out[0]:
                det.append((cur["i"], cx, cy, area))
        return out
    return detectDroplets


def trackInit(self, centre, t):
    origInit(self, centre, t)
    if cur["i"] in want:
        sightings.append((cur["i"], self.id, float(centre[0]), float(centre[1]), 1))


def trackUpdate(self, centre, t, maxSpeed):
    origUpdate(self, centre, t, maxSpeed)
    if cur["i"] in want:
        sightings.append((cur["i"], self.id, float(centre[0]), float(centre[1]),
                          len(self.points)))


def autoROI(fullBackground):
    roi = origROI(fullBackground)
    if roi is not None:
        meta["roi"] = [roi["x"], roi["y"], roi["w"], roi["h"], roi["spacing"]]
    return roi


def main():
    variant = sys.argv[1]
    if variant not in ("classical", "after", "replace"):
        sys.exit(__doc__)
    sample = C.loadJson(os.path.join(C.LABELS, "sample.json"))
    want.update(int(it["frame"]) for it in sample["items"])
    gate = None
    if variant != "classical":
        gate = G.Gate(MODEL)
        G.install(P, gate, variant)
    P.detectDroplets = recordOn(P.detectDroplets)
    S.FramesSource.frames = frames
    P.Track.__init__ = trackInit
    P.Track.update = trackUpdate
    P.autoROI = autoROI
    P._beep = lambda: None
    P.writeLiveStatus = lambda **k: None

    out = os.path.join(C.DATA, "replays")
    t0 = time.perf_counter()
    falls = P.trackAll("frames:" + C.fixture(FIXTURE), show = False, saveVideo = False,
                       outDir = os.path.join(out, "runs", variant))
    minutes = (time.perf_counter() - t0) / 60.0
    d = np.array(det, np.float64).reshape(-1, 4)
    np.savez_compressed(
        os.path.join(out, variant + ".npz"),
        frame = d[:, 0].astype(np.int32), x = d[:, 1].astype(np.float32),
        y = d[:, 2].astype(np.float32), area = d[:, 3].astype(np.float32),
        gate = np.zeros(len(d), np.int8),
        sightings = np.array(sightings, np.float32).reshape(-1, 5),
        roi = np.array(meta.get("roi", [np.nan] * 5), np.float64),
        confirmPoints = np.int32(P.CONFIRM_POINTS))
    summary = {"variant": variant, "minutes": round(minutes, 1),
               "labelledFrames": len(want), "detections": len(d),
               "gateCalls": gate.calls if gate else 0, "falls": len(falls),
               "legalFalls": int(sum(1 for f in falls if f.legal))}
    with open(os.path.join(out, variant + ".json"), "w", encoding = "utf-8") as f:
        json.dump(summary, f, indent = 1)
    print(json.dumps(summary), flush = True)


if __name__ == "__main__":
    main()
