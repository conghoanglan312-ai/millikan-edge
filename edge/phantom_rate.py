"""The phantom rate on the worst-case fixture, with and without the classifier.

    python edge/phantom_rate.py          -> results/fp32/PHANTOM_RATE.md

The same computation as the parent's test/tools/live_check.py: live_20260903_noisy
(471 raw frames from the unit that was noisiest on 2026-09-03), background = the
median of its frames, detectDroplets() per frame, detections per frame averaged.
The bar is live_check's own: 0.87 a frame, measured 2026-09-03.  That fixture is
kept out of training, and its unit is not one the classifier has seen.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv import pipeline as P                                         # noqa: E402
from edge import common as C                                         # noqa: E402
from edge import gate as G                                           # noqa: E402

BAR = 0.87
MODEL = os.path.join(C.RESULTS, "fp32", "model.onnx")


def main():
    fs = sorted(glob.glob(os.path.join(C.fixture("live_20260903_noisy"), "*.png")))
    frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in fs]
    roi = P.autoROI(frames[0])
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    stack = [f[y:y + h, x:x + w] for f in frames]
    bg = np.median(np.stack(stack), axis = 0).astype(np.uint8)
    orig = P.detectDroplets
    rows = []
    for name, mode in (("classical (current code)", None),
                       ("classifier after the gates", "after"),
                       ("classifier replacing area + circularity gates", "replace")):
        P.detectDroplets = orig
        if mode:
            G.install(P, G.Gate(MODEL), mode)
        s = P.buildSettings(roi, roi["spacing"])
        rate = float(np.mean([len(P.detectDroplets(g, bg, s)[0]) for g in stack]))
        rows.append((name, rate))
        print("%-48s %6.2f detections/frame" % (name, rate), flush = True)
    P.detectDroplets = orig
    L = ["# Phantom rate — live_20260903_noisy", "",
         "%d frames, median background, as `live_check.py` computes it. Bar: %.2f a "
         "frame (2026-09-03). Classifier `results/fp32/model.onnx`, threshold %.2f. "
         "Written by `edge/phantom_rate.py`." % (len(stack), BAR, G.THRESHOLD), "",
         "| detector | detections / frame | vs the 0.87 bar |", "|---|---|---|"]
    L += ["| %s | %.2f | %s |" % (n, r, "PASS" if r <= BAR * 2 else "FAIL")
          for n, r in rows]
    L += ["", "PASS/FAIL uses live_check's own tolerance: within 2x the bar."]
    out = os.path.join(C.RESULTS, "fp32", "PHANTOM_RATE.md")
    with open(out, "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("-> %s" % out)


if __name__ == "__main__":
    main()
