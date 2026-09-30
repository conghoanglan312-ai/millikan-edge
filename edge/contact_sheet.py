"""Contact sheets of auto-labelled crops, for a person to eyeball before trusting them.

    python edge/contact_sheet.py panel_sweep3_0909      -> data/autolabel/sheets/*.png

One sheet per class, crops drawn at random (fixed seed): what the tracker confirmed,
what passed every gate but was never confirmed, and what the circularity gate threw
out.  Each tile is the frame minus its background, contrast-stretched, 4x enlarged,
with the 24x24 model window outlined.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402

N = 96                      # tiles per sheet: 8 rows of 12
SCALE = 4


def tile(gray, bg):
    d = cv2.absdiff(gray, bg).astype(np.float32)
    d = np.clip(d * (255.0 / max(8.0, float(d.max()))), 0, 255).astype(np.uint8)
    t = cv2.resize(d, None, fx = SCALE, fy = SCALE, interpolation = cv2.INTER_NEAREST)
    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    a, b = 4 * SCALE, 28 * SCALE - 1
    cv2.rectangle(t, (a, a), (b, b), (0, 160, 255), 1)
    return t


def sheet(z, idx, title):
    rng = np.random.default_rng(20260929)
    pick = rng.choice(idx, min(N, len(idx)), replace = False) if len(idx) else []
    tiles = [tile(z["gray"][k], z["bg"][k]) for k in pick]
    side = 32 * SCALE
    while len(tiles) < N:
        tiles.append(np.zeros((side, side, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 12:(r + 1) * 12]) for r in range(N // 12)])
    head = np.zeros((36, grid.shape[1], 3), np.uint8)
    cv2.putText(head, "%s  (%d of %d shown)" % (title, min(N, len(idx)), len(idx)),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return np.vstack([head, grid])


def main():
    name = sys.argv[1]
    # Read each array once: indexing an NpzFile re-reads and re-inflates the whole
    # array on every access, which made one sheet take minutes.
    with np.load(os.path.join(C.DATA, "autolabel", name + ".npz")) as f:
        z = {k: f[k] for k in ("gray", "bg", "label", "gate", "gates")}
    gates = list(z["gates"])
    out = os.path.join(C.DATA, "autolabel", "sheets")
    os.makedirs(out, exist_ok = True)
    for tag, idx, title in (
            ("1_confirmed", np.where(z["label"] == 1)[0],
             "%s: label 1, sighting of a confirmed track" % name),
            ("2_unconfirmed", np.where(z["label"] == 0)[0],
             "%s: label 0, passed every gate, never confirmed" % name),
            ("3_circularity", np.where(z["gate"] == gates.index("circularity"))[0],
             "%s: rejected by the circularity gate" % name)):
        path = os.path.join(out, "%s_%s.png" % (name, tag))
        cv2.imwrite(path, sheet(z, idx, title))
        print("-> %s" % path)


if __name__ == "__main__":
    main()
