"""The two hottest pieces of the live path, made cheaper without changing an answer.

    from edge import fast; fast.install(P)

Profiled on the board (spray_labelled_0907, window on, INT8 gate): Track.recentSpeed
took 21% of the live path's time.  It is called ~50 times a frame -- the hold check,
the hunt aid, the pick -- often for the same track in the same frame, and each call
tests every one of its 30 points against every grid line in a Python loop.  The hunt
aid's Track.windowFit scans a track's whole history to find its last 3 s.

Same arithmetic, same answers:
  recentSpeed   the grid-line test as one numpy expression (the same float64
                subtraction, abs and comparison), the same np.polyfit; remembered
                per track until the track gains a point -- points are only ever
                appended, so their count is the track's state
  windowFit     the same predicate, now - t <= seconds, found by bisection
                (times only grow along a track, so the predicate is monotonic in
                the index) instead of a scan; the same polyfit on the same points
Installed from outside cv/, which stays untouched, like the classifier gate.

    python edge/fast.py check <fixture> <frames>

replays the first <frames> of a recording through edge/suggestions.py with and
without it and compares every fall, ring, pick and sighting.
"""
import json
import os
import sys

import numpy as np


def install(P):
    Track = P.Track
    grid = P.GRID_BIAS_PX
    fitS = P.CANDIDATE_FIT_S

    def speed(self, spacing, window, rows):
        pts = self.points[-window:]
        if len(pts) < 8:
            return None
        a = np.array(pts, dtype = float)
        if rows:
            far = np.abs(a[:, 1, None] - np.array(rows, dtype = float)[None, :]) \
                .min(axis = 1) > grid
            if int(far.sum()) >= 8:
                a = a[far]
        if a[-1, 0] == a[0, 0]:
            return None
        return float(np.polyfit(a[:, 0], a[:, 1], 1)[0]) / spacing

    def recentSpeed(self, spacing, window = 30):
        rows = getattr(self, "gridRows", None)
        key = (len(self.points), spacing, window, tuple(rows) if rows else None)
        memo = self.__dict__.get("_speedMemo")
        if memo is not None and memo[0] == key:
            return memo[1]
        v = speed(self, spacing, window, rows)
        self._speedMemo = (key, v)
        return v

    def windowFit(self, spacing, now, seconds = None):
        seconds = fitS if seconds is None else seconds
        pts = self.points
        lo, hi = 0, len(pts)
        while lo < hi:                       # first point with now - t <= seconds
            mid = (lo + hi) // 2
            if now - pts[mid][0] <= seconds:
                hi = mid
            else:
                lo = mid + 1
        pts = pts[lo:]
        if len(pts) < 20 or spacing <= 0:
            return None
        a = np.array(pts, dtype = float)
        ts, ys = a[:, 0], a[:, 1]
        if ts[-1] - ts[0] < seconds * 0.6:
            return None
        m, c = np.polyfit(ts, ys, 1)
        resid = float(np.sqrt(np.mean((ys - (m * ts + c)) ** 2)))
        return float(m) / spacing, resid, float(ys[-1] - ys[0]) / spacing

    Track.recentSpeed = recentSpeed
    Track.windowFit = windowFit


def child(fixture, frames, out, useFast):
    """One replay of edge/suggestions.py (classifier after the gates), cut short."""
    from cv import pipeline as P
    from edge import suggestions as SG
    if useFast:
        install(P)
    SG.OUT = out
    orig = P.trackAll
    P.trackAll = lambda *a, **k: orig(*a, toFrame = frames, **k)
    SG.run("after", fixture)


def check(fixture, frames):
    """Replay with and without, then compare every ring, pick, sighting and fall."""
    import glob
    import subprocess
    import tempfile
    root = tempfile.mkdtemp(prefix = "fast_check_")
    runs = []
    for useFast in (0, 1):
        out = os.path.join(root, str(useFast))
        runs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "child",
                                      fixture, str(frames), out, str(useFast)],
                                     stdout = subprocess.DEVNULL))
    for r in runs:
        r.wait()
    same = True
    a, b = [np.load(os.path.join(root, str(k), "%s_after.npz" % fixture)) for k in (0, 1)]
    for key in ("rings", "picks", "sightings"):
        eq = a[key].shape == b[key].shape and np.array_equal(a[key], b[key])
        print("%-10s %s  (%d rows)" % (key, "same" if eq else "DIFFERENT", len(a[key])))
        same &= eq
    fa, fb = [sorted(glob.glob(os.path.join(root, str(k), "runs", fixture, "after", "*falls.csv")))
              for k in (0, 1)]
    for x, y in zip(fa, fb):
        with open(x, encoding = "utf-8") as f, open(y, encoding = "utf-8") as g:
            eq = f.read() == g.read()
        print("%-10s %s" % (os.path.basename(x), "same" if eq else "DIFFERENT"))
        same &= eq
    sa, sb = [json.load(open(os.path.join(root, str(k), "%s_after.json" % fixture)))
              for k in (0, 1)]
    print("replay minutes: %.1f without, %.1f with" % (sa["replayMinutes"], sb["replayMinutes"]))
    print("ALL SAME" if same and fa else "NOT THE SAME", flush = True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1")
    elif sys.argv[1] == "check":
        check(sys.argv[2], int(sys.argv[3]))
