"""Browsable MP4 of a raw-frame fixture, for a person to skim.

    python edge/preview.py one_droplet_slow        -> data/preview/one_droplet_slow.mp4

The fixtures are folders of lossless PNGs, tens of thousands each, which no one can
browse.  This writes every 3rd frame of the ROI at 30 fps (about 4x the capture
rate), contrast-stretched the way the labelling tool shows frames, with the capture
clock burned into the corner so a span can be reported back as "from 612 s to 655 s".
The MP4 is for eyes only -- H.264-style compression destroys droplets for the
pipeline, so nothing is ever measured on it.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402

STEP = 3
STRETCH = np.clip(np.arange(256) * 2.5 - 30, 0, 255).astype(np.uint8)


def main():
    name = sys.argv[1]
    folder = C.fixture(name)
    times = np.load(os.path.join(folder, "times.npy"))
    info = dict(l.split() for l in open(os.path.join(folder, "roi.txt")))
    x, y, w, h = (int(info[k]) for k in ("x", "y", "w", "h"))
    out = os.path.join(C.DATA, "preview")
    os.makedirs(out, exist_ok = True)
    path = os.path.join(out, name + ".mp4")
    writer = None
    for i in range(0, len(times), STEP):
        g = cv2.imread(os.path.join(folder, "f%06d.png" % i), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if name not in C.CROPPED:
            g = g[y:y + h, x:x + w]
        img = cv2.cvtColor(cv2.LUT(g, STRETCH), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (0, 0), (150, 26), (0, 0, 0), -1)
        cv2.putText(img, "%7.1f s" % float(times[i]), (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                     (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    print("-> %s  (%d frames of %d)" % (path, (len(times) + STEP - 1) // STEP, len(times)))


if __name__ == "__main__":
    main()
