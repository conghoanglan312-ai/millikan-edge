"""Time the classical detector, detectDroplets(), per frame -- the classical row's latency.

    python edge/bench_detect.py [--fixtures DIR] [--out FILE]

Two inputs, the same frames on every machine:
  noisy     live_20260903_noisy, 471 raw frames: the worst case this project has seen
            (live_check's phantom fixture), background = the median of its frames,
            exactly as live_check.py builds it
  bench150  the 150 hand-labelled bench_0908 frames, background = the fixture's
            background.png: the typical case

Only detectDroplets() is timed -- threshold, morphology, contours, gates -- because
that is the stage a classifier replaces.  Frame decode and ROI crop are outside the
timer.  The first WARMUP frames are dropped.  Reported: median, p90 and mean ms per
frame, and the FPS the mean allows.
"""
import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2                                                           # noqa: E402
from cv import pipeline as P                                         # noqa: E402

WARMUP = 20


def roiStack(frames, roi):
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    return [f[y:y + h, x:x + w] for f in frames]


def timeIt(stack, bg, s):
    times, dets = [], []
    for k, g in enumerate(stack):
        t0 = time.perf_counter()
        out = P.detectDroplets(g, bg, s)
        dt = time.perf_counter() - t0
        if k >= WARMUP:
            times.append(dt * 1000.0)
            dets.append(len(out[0]))
    a = np.array(times)
    return {"frames": len(a), "median_ms": round(float(np.median(a)), 2),
            "p90_ms": round(float(np.percentile(a, 90)), 2),
            "mean_ms": round(float(a.mean()), 2),
            "fps": round(1000.0 / float(a.mean()), 1),
            "detections_per_frame": round(float(np.mean(dets)), 2)}


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default = os.path.join(
        os.environ.get("MILLIKAN_PARENT",
                       r"C:\Users\LAN\Documents\GitHub\Physics-Experiment-Assistant"),
        "test", "fixtures"))
    ap.add_argument("--labels", default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cv", "for_report", "detection_accuracy", "sample.json"))
    ap.add_argument("--out", default = None)
    a = ap.parse_args()
    res = {"machine": platform.machine(), "node": platform.node(),
           "python": platform.python_version(), "opencv": cv2.__version__,
           "numpy": np.__version__, "cv2_threads": cv2.getNumThreads()}

    fs = sorted(glob.glob(os.path.join(a.fixtures, "live_20260903_noisy", "*.png")))
    frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]
    roi = P.autoROI(frames[0])
    s = P.buildSettings(roi, roi["spacing"])
    stack = roiStack(frames, roi)
    bg = np.median(np.stack(stack), axis = 0).astype(np.uint8)
    res["noisy"] = timeIt(stack, bg, s)

    bench = os.path.join(a.fixtures, "bench_0908")
    with open(a.labels, encoding = "utf-8") as f:
        want = sorted({int(it["frame"]) for it in json.load(f)["items"]})
    paths = [os.path.join(bench, "f%06d.png" % i) for i in want]
    paths = [p for p in paths if os.path.exists(p)]
    if paths:
        full = cv2.imread(os.path.join(bench, "background.png"), cv2.IMREAD_GRAYSCALE)
        roi = P.autoROI(full)
        s = P.buildSettings(roi, roi["spacing"])
        frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        res["bench150"] = timeIt(roiStack(frames, roi), roiStack([full], roi)[0], s)
    text = json.dumps(res, indent = 1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding = "utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
