"""Build the training and validation sets from the auto-labelled crops.

    python edge/dataset.py        -> data/dataset/train.npz, val.npz, SUMMARY.md

The rules are the owner's decision of 2026-10-09 (HANDOVER §11), set on a blind
review of 236 tracks (data/review/REPORT.md):

  positive   spray_labelled_0907: every confirmed track
             unit2-4_1009, one_droplet_slow, one_droplet_full: confirmed tracks of
             >= 200 sightings
  negative   phantom tracks the review exposed: zoom_labelled_0907 confirmed < 200,
             one_droplet_slow confirmed < 50
             every blob the detector kept in a droplet-free sweep
             blobs the area and circularity gates rejected -- the gates the
             classifier replaces.  The position masks (grid lines, panel margin,
             static marks, frame edge) stay classical, so their rejects are not
             the classifier's problem and are left out
  dropped    everything else, in particular kept-but-unconfirmed blobs from
             droplet sessions (a third of them are droplets on bench_0908)

Split by recording day: train 10-09, 09-05, 09-09; validate 09-07, 09-08 evening.
bench_0908 (09-08 morning) is the test set and never appears here.

Crops of one track are near-duplicates, so a track gives at most PER_TRACK crops,
spread over its life; gate rejects are sampled to at most PER_GATE per recording
and gate.  Balancing recordings against each other happens at training time.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge import common as C                                         # noqa: E402

PER_TRACK = 20
PER_GATE = 5000
SHAPE_GATES = ("area<min", "area>max", "circularity")

SPLIT = {"unit2-4_1009": "train", "one_droplet_slow": "train",
         "one_droplet_full": "train", "panel_sweep3_0909": "train",
         "spray_labelled_0907": "val", "zoom_labelled_0907": "val",
         "voltage_sweep_0907": "val", "panel_sweep_0908": "val",
         "panel_sweep2_0908": "val"}
POSITIVE_ALL = {"spray_labelled_0907"}
POSITIVE_LONG = {"unit2-4_1009", "one_droplet_slow", "one_droplet_full"}
PHANTOM_BELOW = {"zoom_labelled_0907": 200, "one_droplet_slow": 50}
SWEEPS = {"panel_sweep_0908", "panel_sweep2_0908", "panel_sweep3_0909",
          "voltage_sweep_0907"}
LONG = 200

# why a crop is in the set -- kept with it, so a result can be broken down by source
KINDS = ("droplet", "phantom track", "sweep blob", "gate reject")


def perTrack(idx, track, rng):
    """At most PER_TRACK crops per track, evenly spread over its life."""
    out = []
    for t in np.unique(track[idx]):
        rows = idx[track[idx] == t]
        if len(rows) > PER_TRACK:
            rows = rows[np.linspace(0, len(rows) - 1, PER_TRACK).astype(int)]
        out.append(rows)
    return np.concatenate(out) if out else np.array([], int)


def select(name, z, rng):
    lab, tp, gate, track = z["label"], z["trackPoints"], z["gate"], z["track"]
    gates = [str(g) for g in z["gates"]]
    picks = []                                      # (rows, y, kind)
    if name in POSITIVE_ALL:
        picks.append((perTrack(np.where(lab == 1)[0], track, rng), 1, 0))
    if name in POSITIVE_LONG:
        picks.append((perTrack(np.where((lab == 1) & (tp >= LONG))[0], track, rng), 1, 0))
    if name in PHANTOM_BELOW:
        rows = np.where((lab == 1) & (tp < PHANTOM_BELOW[name]))[0]
        picks.append((perTrack(rows, track, rng), 0, 1))
    if name in SWEEPS:
        picks.append((np.where(gate == 0)[0], 0, 2))
    for g in SHAPE_GATES:
        rows = np.where(gate == gates.index(g))[0]
        if len(rows) > PER_GATE:
            rows = np.sort(rng.choice(rows, PER_GATE, replace = False))
        picks.append((rows, 0, 3))
    return picks


def main():
    rng = np.random.default_rng(20261009)
    out = os.path.join(C.DATA, "dataset")
    os.makedirs(out, exist_ok = True)
    parts = {"train": [], "val": []}
    table = []
    for name, split in SPLIT.items():
        with np.load(os.path.join(C.DATA, "autolabel", name + ".npz")) as f:
            z = {k: f[k] for k in ("gray", "bg", "label", "trackPoints", "gate",
                                   "track", "gates", "frame")}
        for rows, y, kind in select(name, z, rng):
            if not len(rows):
                continue
            parts[split].append({
                "gray": z["gray"][rows], "bg": z["bg"][rows],
                "y": np.full(len(rows), y, np.int8),
                "kind": np.full(len(rows), kind, np.int8),
                "rec": np.full(len(rows), list(SPLIT).index(name), np.int16),
                "track": z["track"][rows], "frame": z["frame"][rows]})
            table.append((split, name, KINDS[kind], y, len(rows)))
    summary = {"recordings": list(SPLIT), "kinds": list(KINDS),
               "perTrack": PER_TRACK, "perGate": PER_GATE}
    for split, items in parts.items():
        cat = {k: np.concatenate([p[k] for p in items]) for k in items[0]}
        np.savez_compressed(os.path.join(out, split + ".npz"), **cat,
                            recordings = np.array(list(SPLIT)),
                            kinds = np.array(KINDS))
        summary[split] = {"crops": int(len(cat["y"])), "droplet": int(cat["y"].sum()),
                          "noise": int((cat["y"] == 0).sum())}
    L = ["# Training data — built by `edge/dataset.py`", "",
         "| split | recording | kind | label | crops |", "|---|---|---|---|---|"]
    L += ["| %s | %s | %s | %s | %d |" % (s, n, k, "droplet" if y else "noise", c)
          for s, n, k, y, c in table]
    L += ["", "| split | crops | droplet | noise |", "|---|---|---|---|"]
    L += ["| %s | %d | %d | %d |" % (s, summary[s]["crops"], summary[s]["droplet"],
                                     summary[s]["noise"]) for s in ("train", "val")]
    with open(os.path.join(out, "SUMMARY.md"), "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding = "utf-8") as f:
        json.dump(summary, f, indent = 1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
