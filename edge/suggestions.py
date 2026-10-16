"""What the hunt aid points the operator at, with and without the classifier.

The classifier's gain so far is counted in detections, and the operator never sees a
detection.  They see the hunt aid: rings on the droplets candidates() thinks worth
their attention, and the square it puts on a droplet when nobody has clicked for
IDLE_PICK_S.  This asks whether the false detections the classifier removes ever
reached those -- whether fewer phantoms means fewer wrong suggestions.

Fixed before any replay ran (2026-10-16):
  suggestion   a track ringed by candidates() or squared by the idle auto-pick
  wrong        a suggested track a person judges not a droplet, from a blind clip of
               it (edge/review_tracks.py's clips; session and variant hidden)
  headline     wrong suggestions per 10 minutes of session, and the share of
               suggested tracks that are wrong: classical vs the classifier after the
               gates (FP32; INT8 makes the same decision on 99.9%)
  sessions     bench_0908 (the test session) and the two unseen units -- never a
               training or validation recording
Every number goes in the report, whichever way it points.

    python edge/suggestions.py run classical|after <fixture>   one replay
    python edge/suggestions.py clips      blind clips of every suggested track
    python edge/suggestions.py review     y droplet / n not / u unsure / b back / q quit
    python edge/suggestions.py report     -> results/suggestions/

The replay runs the pipeline's display path (show = True) with the window calls
taken over, as edge/live_board.py does: candidates() and the auto-pick only run
there.  Nobody clicks, so the auto-pick fires whenever it may.  The rings counted
are candidates()'s output; the display can still hide them later (while a fall is
being measured), the same way in both variants.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv import pipeline as P                                         # noqa: E402
from cv import source as S                                           # noqa: E402
from edge import common as C                                         # noqa: E402
from edge import gate as G                                           # noqa: E402
from edge import review_tracks as RT                                 # noqa: E402

FIXTURES = ("bench_0908", "unit1-3_1015", "unit3-2_1015")
VARIANTS = ("classical", "after")
MODEL = os.path.join(C.RESULTS, "fp32", "model.onnx")
OUT = os.path.join(C.DATA, "suggestions")
CLIPS = os.path.join(OUT, "clips.npz")
ANSWERS = os.path.join(OUT, "answers.json")
REPORT = os.path.join(C.RESULTS, "suggestions")
SAME_PX = 3.0           # two runs' tracks are one object if most of their shared
SAME_SHARE = 0.5        # frames put them this close

cur = {"i": -1, "t0": None, "t1": None, "calls": 0}
rings, picks, sightings, meta = [], [], [], {}
origFrames = S.FramesSource.frames
origInit, origUpdate = P.Track.__init__, P.Track.update
origCandidates, origSame, origROI = P.candidates, P.samePickedDroplet, P.autoROI


class Headless(object):
    """cv2 inside the pipeline with no window: drawing runs, showing does nothing."""

    def __getattr__(self, name):
        return getattr(cv2, name)

    def namedWindow(self, *a, **k):
        pass

    def setMouseCallback(self, *a, **k):
        pass

    def imshow(self, *a, **k):
        pass

    def destroyAllWindows(self, *a, **k):
        pass

    def waitKey(self, *a, **k):
        return -1                   # no key, and no waiting to keep real time


def frames(self, first = 0, last = None):
    j = first
    for frame, t in origFrames(self, first, last):
        while j < len(self.times) and float(self.times[j]) != t:
            j += 1
        cur["i"] = j
        cur["t0"] = t if cur["t0"] is None else cur["t0"]
        cur["t1"] = t
        yield frame, t
        j += 1


def candidates(*a, **k):
    out = origCandidates(*a, **k)
    cur["calls"] += 1
    for rank, c in enumerate(out):
        rings.append((cur["i"], c["id"], c["pos"][0], c["pos"][1], rank,
                      {True: 1, False: 0, None: -1}[c["charged"]]))
    return out


def samePickedDroplet(spot, t, now):
    """Called when the auto-pick squares a track (and on a click, which a replay
    never has)."""
    picks.append((cur["i"], t.id, float(t.pos[0]), float(t.pos[1])))
    return origSame(spot, t, now)


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


def runPath(fixture, variant, ext):
    return os.path.join(OUT, "%s_%s.%s" % (fixture, variant, ext))


def run(variant, fixture):
    if variant not in VARIANTS or fixture not in FIXTURES:
        sys.exit(__doc__)
    P.cv2 = Headless()
    gate = None
    if variant == "after":
        gate = G.Gate(MODEL)
        G.install(P, gate, "after")
    P.candidates = candidates
    P.samePickedDroplet = samePickedDroplet
    S.FramesSource.frames = frames
    P.Track.__init__ = trackInit
    P.Track.update = trackUpdate
    P.autoROI = autoROI
    P._beep = lambda: None
    P.writeLiveStatus = lambda **k: None
    P.publishFrame = lambda *a, **k: None
    t0 = time.perf_counter()
    falls = P.trackAll("frames:" + C.fixture(fixture), show = True, saveVideo = False,
                       outDir = os.path.join(OUT, "runs", fixture, variant))
    r = np.array(rings, np.float64).reshape(-1, 6)
    p = np.array(picks, np.float64).reshape(-1, 4)
    s = np.array(sightings, np.float64).reshape(-1, 5)
    ids = np.union1d(r[:, 1], p[:, 1])
    os.makedirs(OUT, exist_ok = True)
    np.savez_compressed(runPath(fixture, variant, "npz"), rings = r, picks = p,
                        sightings = s[np.isin(s[:, 1], ids)],
                        roi = np.array(meta.get("roi", [np.nan] * 5), np.float64))
    summary = {"fixture": fixture, "variant": variant,
               "replayMinutes": round((time.perf_counter() - t0) / 60.0, 1),
               "sessionMinutes": round((cur["t1"] - cur["t0"]) / 60.0, 2),
               "evaluations": cur["calls"],
               "ringedTracks": int(len(np.unique(r[:, 1]))),
               "pickedTracks": int(len(np.unique(p[:, 1]))),
               "gateCalls": gate.calls if gate else 0, "falls": len(falls),
               "legalFalls": int(sum(1 for f in falls if f.legal))}
    with open(runPath(fixture, variant, "json"), "w", encoding = "utf-8") as f:
        json.dump(summary, f, indent = 1)
    print(json.dumps(summary), flush = True)


def loadRun(fixture, variant):
    with np.load(runPath(fixture, variant, "npz")) as z:
        d = {k: z[k] for k in z.files}
    d["summary"] = C.loadJson(runPath(fixture, variant, "json"))
    s = d["sightings"]
    d["tracks"] = {int(t): s[s[:, 1] == t] for t in np.unique(s[:, 1])}
    d["suggested"] = sorted(int(t) for t in np.union1d(d["rings"][:, 1], d["picks"][:, 1]))
    return d


def same(a, b):
    """Most of the frames the shorter track shares with the other put them together."""
    at = {int(r[0]): (r[2], r[3]) for r in a}
    hits = sum(1 for r in b if int(r[0]) in at
               and (at[int(r[0])][0] - r[2]) ** 2 + (at[int(r[0])][1] - r[3]) ** 2
               <= SAME_PX ** 2)
    return hits >= SAME_SHARE * min(len(a), len(b))


def objects(fixture):
    """The suggested tracks of both variants, one entry per thing on the screen."""
    runs = {v: loadRun(fixture, v) for v in VARIANTS}
    out = []
    for v in VARIANTS:
        for t in runs[v]["suggested"]:
            rows = runs[v]["tracks"][t]
            hit = next((o for o in out if o["variant0"] != v
                        and same(o["rows"], rows)), None)
            if hit is None:
                out.append({"variant0": v, "rows": rows, "members": [[v, t]]})
            else:
                hit["members"].append([v, t])
                if len(rows) > len(hit["rows"]):
                    hit["rows"] = rows
    return out, runs


def clips():
    rng = np.random.default_rng(20261016)
    cl, mt = [], []
    for fixture in FIXTURES:
        if not all(os.path.exists(runPath(fixture, v, "npz")) for v in VARIANTS):
            print("%s: both replays needed -- skipped" % fixture)
            continue
        objs, runs = objects(fixture)
        roi = runs["classical"]["roi"]
        for o in objs:
            cl.append(RT.clip(fixture, o["rows"], roi))
            mt.append({"fixture": fixture, "members": o["members"]})
        print("%-14s %d suggested objects" % (fixture, len(objs)), flush = True)
    order = rng.permutation(len(cl))
    np.savez_compressed(CLIPS, clips = np.array(cl)[order],
                        meta = np.array([json.dumps(mt[k]) for k in order]))
    print("-> %s  (%d clips)" % (CLIPS, len(cl)))


def report():
    _clips, mt = RT.load(CLIPS)
    answers = C.loadJson(ANSWERS, {})
    verdict = {}
    for k, m in enumerate(mt):
        a = answers.get(str(k), {}).get("a")
        for v, t in m["members"]:
            verdict[(m["fixture"], v, t)] = a
    L = ["# What the hunt aid suggests, with and without the classifier", "",
         "Suggested = ringed by `candidates()` or squared by the idle auto-pick, in a "
         "replay with nobody clicking. Each suggested track judged by a person from a "
         "blind clip (session and variant hidden). Written by `edge/suggestions.py`.", "",
         "| session | minutes | variant | suggested tracks | droplet | not | unsure | "
         "not a droplet, share | wrong / 10 min | top ring on a non-droplet |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for fixture in FIXTURES:
        if not all(os.path.exists(runPath(fixture, v, "npz")) for v in VARIANTS):
            continue
        for v in VARIANTS:
            d = loadRun(fixture, v)
            mins = d["summary"]["sessionMinutes"]
            got = [verdict.get((fixture, v, t)) for t in d["suggested"]]
            y, n, u = got.count("y"), got.count("n"), got.count("u")
            top = d["rings"][d["rings"][:, 4] == 0]
            topWrong = sum(1 for r in top if verdict.get((fixture, v, int(r[1]))) == "n")
            share = n / float(y + n) if y + n else float("nan")
            L.append("| %s | %.1f | %s | %d | %d | %d | %d | %.0f%% | %.1f | %.0f%% of %d |"
                     % (fixture, mins, v, len(got), y, n, u, 100 * share,
                        10.0 * n / mins, 100.0 * topWrong / max(1, len(top)), len(top)))
            out.setdefault(fixture, {})[v] = {
                "minutes": mins, "suggested": len(got), "droplet": y, "not": n,
                "unsure": u, "unanswered": got.count(None), "wrongShare": share,
                "wrongPer10Min": 10.0 * n / mins, "topRings": len(top),
                "topRingsWrong": topWrong, **d["summary"]}
    os.makedirs(REPORT, exist_ok = True)
    with open(os.path.join(REPORT, "REPORT.md"), "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT, "report.json"), "w", encoding = "utf-8") as f:
        json.dump(out, f, indent = 1)
    print("\n".join(L))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run" and len(sys.argv) == 4:
        run(sys.argv[2], sys.argv[3])
    elif cmd == "clips":
        clips()
    elif cmd == "review":
        RT.review(CLIPS, ANSWERS)
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
