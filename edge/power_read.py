"""The board's watts, read from a film of the bench supply's screen.

    python edge/power_read.py frames <video> <folder>    a frame every 2 s, enlarged
    powershell -File edge/power_ocr.ps1 -Dir <folder> -Out <ocr.txt>     Windows' OCR
    python edge/power_read.py parse results/power/board_ocr_20261029.txt [out.csv]

Each frame's text gives the first watts and amps it holds, which are channel 1's.
Readings under 0.5 W with the board on are misreads, the idle channel's 000.00 mostly,
and are dropped. The phases were read off the trace of the 2026-10-29 run: boot, idle,
the 10-minute run (from 20 s after it started, the laptop's cable out), idle again.
"""
import csv
import re
import sys

PHASES = (("idle before", 30, 136), ("test", 160, 736), ("idle after", 746, 830))


def frames(video, folder):
    import os

    import cv2
    os.makedirs(folder, exist_ok = True)
    c = cv2.VideoCapture(video)
    fps = c.get(cv2.CAP_PROP_FPS)
    for f in range(0, int(c.get(cv2.CAP_PROP_FRAME_COUNT)), int(round(fps * 2))):
        c.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = c.read()
        if ok:
            cv2.imwrite(os.path.join(folder, "%05.0f.png" % (f / fps)),
                        cv2.resize(im[120:320, :470], None, fx = 2, fy = 2,
                                   interpolation = cv2.INTER_CUBIC))


def phase(t):
    for name, a, b in PHASES:
        if a <= t < b:
            return name
    return "boot / switching"


def parse(path):
    rows = []
    for line in open(path, encoding = "utf-8-sig"):
        name, _, text = line.rstrip("\n").partition("\t")
        m = re.search(r"(?<![\d.])0{0,2}(\d)[.,]?(\d{2})[.,wW]", text)
        a = re.search(r"(?<!\d)0[.,](\d{4})", text)
        w = int(m.group(1)) + int(m.group(2)) / 100 if m else ""
        ph = phase(int(name))
        rows.append([int(name), ph, w, int(a.group(1)) / 1e4 if a else "",
                     int(ph != "boot / switching" and w != "" and w >= 0.5)])
    return rows


def main():
    if sys.argv[1] == "frames":
        frames(sys.argv[2], sys.argv[3])
        return
    rows = parse(sys.argv[2])
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w", newline = "") as f:
            wr = csv.writer(f)
            wr.writerow(["video_s", "phase", "watts_read", "amps_read", "used"])
            wr.writerows(rows)
    for name, _a, _b in PHASES:
        v = [r[2] for r in rows if r[1] == name and r[4]]
        print("%-12s %3d readings, mean %.2f W" % (name, len(v), sum(v) / len(v)))


if __name__ == "__main__":
    main()
