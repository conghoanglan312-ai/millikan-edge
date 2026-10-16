"""Run the live pipeline on a looped recording for a fixed time, window and all, and
log what the machine does meanwhile.

    board:   PYTHONPATH=/opt/cvgui python3 edge/soak.py --minutes 15         (INT8, NPU)
    laptop:  python edge/soak.py --minutes 10 --gate fp32                     (FP32, CPU)

Two uses.  Heat: does the board slow down once it is hot -- every thermal zone, the
CPU and NPU clocks and the frame rate, every 5 s.  Power: the same work for the same
time on either machine, with a plug-in meter between the wall and its adapter.

The recording is spray_labelled_0907 (68 s, ~26 candidate blobs a frame -- the
busiest one the board holds), played round and round: its clock runs on across the
joins, so the pipeline sees one long session in which the scene changes every 68 s.
The window opens as in a real session (edge/live_board.py's Screen); over adb on
the board set DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv import pipeline as P                                         # noqa: E402
from cv import source as S                                           # noqa: E402
from edge import fast as F                                           # noqa: E402
from edge import gate as G                                           # noqa: E402
from edge import live_board as LB                                    # noqa: E402

BOARD_SOURCE = "/opt/fixtures/spray_labelled_0907"
EVERY_S = 5.0


class Timed(LB.Screen):
    """The session's window, closed by the clock instead of q."""

    def __init__(self, seconds):
        LB.Screen.__init__(self)
        self.end = time.perf_counter() + seconds

    def waitKey(self, delay = 0):
        key = LB.Screen.waitKey(self, delay)
        return ord("q") if time.perf_counter() >= self.end else key


def looped(orig):
    def frames(self, first = 0, last = None):
        t = np.asarray(self.times, float)
        period = t[-1] - t[0] + float(np.median(np.diff(t)))
        lap = 0
        while True:
            for frame, ts in orig(self, first, last):
                yield frame, ts + lap * period
            lap += 1
    return frames


def sensors():
    """Every thermal zone in degrees, CPU and NPU clocks in MHz; {} off the board."""
    out = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        with open(z + "/type") as f, open(z + "/temp") as g:
            out[f.read().strip().replace("-thermal", "")] = int(g.read()) / 1000.0
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        with open(p + "/scaling_cur_freq") as f:
            out["cpu" + p[-1] + "_MHz"] = int(f.read()) / 1000.0
    for d in glob.glob("/sys/class/devfreq/*.npu"):
        with open(d + "/cur_freq") as f:
            out["npu_MHz"] = int(f.read()) / 1e6
    return out


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type = float, default = 15.0)
    ap.add_argument("--gate", choices = ("npu", "fp32", "none"), default = "npu")
    ap.add_argument("--source", default = None)
    ap.add_argument("--out", default = None)
    a = ap.parse_args()
    if a.source is None:
        if os.path.isdir(BOARD_SOURCE):
            a.source = BOARD_SOURCE
        else:
            from edge import common as C
            a.source = C.fixture("spray_labelled_0907")
    out = a.out or os.path.join(LB.REPO, "data", "soak")
    os.makedirs(out, exist_ok = True)

    screen = Timed(a.minutes * 60.0)
    P.cv2 = screen
    if a.gate == "npu":
        G.install(P, G.BatchedNpuGate({b: os.path.join(LB.REPO, "results", "int8", f)
                                       for b, f in LB.RKNN.items()}), "after")
    elif a.gate == "fp32":
        G.install(P, G.Gate(os.path.join(LB.REPO, "results", "fp32", "model.onnx")),
                  "after")
    S.FramesSource.frames = looped(S.FramesSource.frames)
    F.install(P)                 # as edge/live_board.py does
    P.SHOW_EVERY = 1
    P._beep = lambda: None
    P.writeLiveStatus = lambda **k: None
    P.publishFrame = lambda *a, **k: None

    log, start = [], time.perf_counter()
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            log.append({"s": round(time.perf_counter() - start, 1),
                        "shown": len(screen.shown), **sensors()})
            stop.wait(EVERY_S)
    threading.Thread(target = watch, daemon = True).start()
    P.trackAll("frames:" + a.source, show = True, saveVideo = False,
               outDir = os.path.join(out, "session"))
    stop.set()

    shown = np.array(screen.shown) - start
    rows = []
    for m in range(int(np.ceil(a.minutes))):
        k = (shown >= 60 * m) & (shown < 60 * (m + 1))
        s = [r for r in log if 60 * m <= r["s"] < 60 * (m + 1)]
        temps = [max(v for key, v in r.items() if not key.endswith("MHz")
                     and key not in ("s", "shown")) for r in s if len(r) > 2]
        span = min(60.0, a.minutes * 60.0 - 60 * m)
        rows.append({"minute": m + 1, "updatesPerSecond": float(k.sum()) / span,
                     "hottestC": max(temps) if temps else None,
                     "bigCoreMHzMin": min((r.get("cpu4_MHz", np.nan) for r in s),
                                          default = None),
                     "npuMHzMin": min((r.get("npu_MHz", np.nan) for r in s),
                                      default = None)})
    name = "soak_%s_%s.json" % (a.gate, time.strftime("%Y%m%d_%H%M%S"))
    with open(os.path.join(out, name), "w", encoding = "utf-8") as f:
        json.dump({"minutes": a.minutes, "gate": a.gate, "source": a.source,
                   "window": screen.smoothness(), "perMinute": rows, "log": log}, f,
                  indent = 1)
    print("minute  updates/s  hottest C  big-core MHz (min)  NPU MHz (min)")
    for r in rows:
        print("%6d  %9.1f  %9s  %18s  %13s" % (
            r["minute"], r["updatesPerSecond"],
            "%.1f" % r["hottestC"] if r["hottestC"] else "-",
            "%.0f" % r["bigCoreMHzMin"] if r["bigCoreMHzMin"] == r["bigCoreMHzMin"]
            and r["bigCoreMHzMin"] is not None else "-",
            "%.0f" % r["npuMHzMin"] if r["npuMHzMin"] == r["npuMHzMin"]
            and r["npuMHzMin"] is not None else "-"))
    print("window: %s" % screen.smoothness())
    print("-> %s" % os.path.join(out, name), flush = True)


if __name__ == "__main__":
    main()
