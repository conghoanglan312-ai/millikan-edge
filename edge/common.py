"""Paths, the fixture registry, and label matching, shared by the edge tools.

The raw frames live in the parent clone's test/fixtures (gitignored there, absent
here).  Point MILLIKAN_PARENT at it when running from WSL2:

    export MILLIKAN_PARENT=/mnt/c/Users/LAN/Documents/GitHub/Physics-Experiment-Assistant
"""
import json
import os

import numpy as np

EDGE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EDGE)
PARENT = os.environ.get("MILLIKAN_PARENT",
                        r"C:\Users\LAN\Documents\GitHub\Physics-Experiment-Assistant")
FIXTURES = os.path.join(PARENT, "test", "fixtures")
DATA = os.path.join(REPO, "data")                   # gitignored: extracted crops
RESULTS = os.path.join(REPO, "results")             # tracked: every published number
LABELS = os.path.join(REPO, "cv", "for_report", "detection_accuracy")

# Every fixture that holds RAW frames (lossless PNG, no overlay), with when it was
# captured -- read from the frame files' timestamps on 2026-09-29.  The recorded
# sessions under test/output hold only annotated video, whose rings mark exactly the
# droplets the tracker found; a crop from it carries its own label, so none of it is
# usable for training.
#
#   role "test"  hand-labelled; never trained on, never tuned on
#   role "eval"  the phantom-rate fixture behind live_check's 3.11 vs 0.87; kept out
#                of training so that number stays an honest before/after
#   role "pool"  available for training/validation; the split is decided by date
FIXTURE_INFO = {
    "bench_0908":          ("2026-09-08 11:51", "test"),
    "live_20260903_noisy": ("2026-09-03 16:43", "eval"),
    "one_droplet_slow":    ("2026-09-05 15:03", "pool"),
    "one_droplet_full":    ("2026-09-05 15:30", "pool"),
    "spray_labelled_0907": ("2026-09-07 11:43", "pool"),
    "zoom_labelled_0907":  ("2026-09-07 12:28", "pool"),
    "voltage_sweep_0907":  ("2026-09-07 17:08", "pool"),
    "panel_sweep_0908":    ("2026-09-08 17:33", "pool"),
    "panel_sweep2_0908":   ("2026-09-08 21:25", "pool"),
    "panel_sweep3_0909":   ("2026-09-09 17:04", "pool"),
    # From 2026-10-09 every recording names its unit (HANDOVER §4): row from the
    # professor's seat, then position left to right as seen from it.
    "unit2-4_1009":        ("2026-10-09 16:51", "pool"),     # unit 2-4, sprays, field off
}

# Saved already ROI-cropped (a bug in the first raw_capture.py), so the ROI comes from
# roi.txt rather than autoROI -- the same override the parent's check.py applies.
CROPPED = {"one_droplet_slow"}

# The parent's acceptance gate for one_droplet_slow (check.py, re-baselined
# 2026-09-09): these five kinematically legal falls.  A replay that wraps the pipeline
# must give them back, or the wrapping changed what it records.
REFERENCE_FALLS = {"one_droplet_slow": [33.44, 33.47, 34.91, 34.95, 35.41]}

# A prediction within this of a label is the same droplet.  The parent's figure
# (detacc_common.MATCH_PX): a droplet blob is ~7 px across, median area 42 px^2.
MATCH_PX = 8.0

# Session phases of bench_0908 on its capture clock, from the parent's
# detacc_common.py.  Overall figures weight each phase by its share of session time.
SPRAYS = (81.0, 958.0, 1750.0)
SPRAY_S = 70.0
MEASURING = ((588.4, 898.4), (1401.2, 1716.3), (2141.0, 2440.5))
STRATA = ("spray", "hunting", "measuring", "quiet")


def fixture(name):
    return os.path.join(FIXTURES, name)


def stratum(t):
    for a, b in MEASURING:
        if a <= t <= b:
            return "measuring"
    for sp, (a, _b) in zip(SPRAYS, MEASURING):
        if sp <= t < sp + SPRAY_S:
            return "spray"
        if sp + SPRAY_S <= t < a:
            return "hunting"
    return "quiet"


def loadJson(path, default = None):
    if not os.path.exists(path):
        return default
    with open(path, encoding = "utf-8") as f:
        return json.load(f)


def saveJson(path, obj):
    """Write via a temporary file, so a crash mid-write never leaves half a label file."""
    os.makedirs(os.path.dirname(path), exist_ok = True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding = "utf-8") as f:
        json.dump(obj, f, indent = 1)
    os.replace(tmp, path)


def _pairs(a, b, radius):
    """Maximum one-to-one matching within `radius`, nearest first among equals."""
    from scipy.optimize import linear_sum_assignment
    if not len(a) or not len(b):
        return np.array([], int), np.array([], int)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis = 2))
    cost = np.where(d <= radius, d, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e6
    return r[ok], c[ok]


def match(pred, sure, unsure, radius = MATCH_PX):
    """TP / FP / FN of predictions against sure labels -- the parent's rule, unchanged.

    Unsure labels are neither rewarded nor penalised: a prediction that lands on one is
    not a false positive, and an unsure label nothing lands on is not a miss.
    """
    pred = np.asarray(pred, float).reshape(-1, 2)
    sure = np.asarray(sure, float).reshape(-1, 2)
    unsure = np.asarray(unsure, float).reshape(-1, 2)
    tpP, tpS = _pairs(pred, sure, radius)
    rest = np.setdiff1d(np.arange(len(pred)), tpP)
    ignored = np.array([], int)
    if len(rest) and len(unsure):
        r, _c = _pairs(pred[rest], unsure, radius)
        ignored = rest[r]
    fp = np.setdiff1d(rest, ignored)
    fn = np.setdiff1d(np.arange(len(sure)), tpS)
    return {"tp": len(tpP), "fp": len(fp), "fn": len(fn),
            "tpP": tpP, "tpS": tpS, "fpIdx": fp, "fnIdx": fn}
