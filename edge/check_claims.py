"""Does what the README says still match the files that produced it?

    python edge/check_claims.py

Each claim names a phrase in README.md and the phrase in a result file it rests on.
Wordings that once went further than the data are listed too, and must stay out.
Run it before anything is published or sent; a failure means the words have drifted.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = "results/fp32_v2/TEST_SCORE_bench_0908.md"
UNSEEN = "results/unseen/SCORE.md"
MJPG = "results/mjpg/SCORE.md"
BOARD = "results/board/README.md"
LIVE = "results/live/README.md"
POWER = "results/power/README.md"

CLAIMS = (
    ("| classical | 81.0% | 95.8% | 87.8% | 1.35 |", TEST,
     "| classical, current code — detector | 81.0% (72.0–88.3) | 95.8% (94.1–97.3) | 87.8% (82.2–92.0) | 1.35 |"),
    ("| + classifier, FP32 | 92.6% | 93.3% | 93.0% | **0.45** |", TEST,
     "| FP32 classifier after the gates — detector | 92.6% (90.4–94.7) | 93.3% (91.3–95.4) | 93.0% (91.3–94.6) | 0.45 |"),
    ("| + classifier, INT8 | 92.1% | 93.7% | 92.9% | 0.48 |", TEST,
     "offline filter — detector | 92.1% (89.8–94.3) | 93.7% (91.7–95.6) | 92.9% (91.2–94.5) | 0.48 |"),
    ("all 19 legal falls", "results/fp32/README.md", "keeps all 19 legal falls"),
    ("99.90% of 41,038", BOARD, "same decision on 99.90%"),
    ("| classical | 69.8% | 87.6% | 77.7% | 11.7 |", UNSEEN,
     "| all unseen | classical detector | 69.8% (65.9–73.6) | 87.6% (86.5–88.7) | 77.7% (75.2–80.1) | 11.66 |"),
    ("| + classifier, INT8 | 93.5% | 87.4% | 90.3% | **1.9** |", UNSEEN,
     "(RKNN simulator) — detector | 93.5% (90.6–95.9) | 87.4% (86.2–88.5) | 90.3% (88.8–91.6) | 1.86 |"),
    ("from 21.7 to 2.2", UNSEEN, "| unit3-2_1015 | classical detector | 57.3% (52.6–62.4) | 86.5% (84.7–88.3) | 68.9% (65.5–72.4) | 21.65 |"),
    ("precision\n84.6% → 37.5%", MJPG, "| unit3-2_1027_A | classical detector | 37.5% (29.9–46.6) | 79.2% (76.6–82.0) | 50.9% (43.6–58.5) | 11.47 |"),
    ("F1 85.4% raw, 85.1% MJPG", MJPG, "(RKNN simulator) — detector | 97.9% (96.8–99.0) | 75.3% (72.6–78.4) | 85.1% (83.3–87.2) | 0.14 |"),
    ("recall 78.7% and 75.3%", MJPG, "(RKNN simulator) — detector | 93.3% (86.5–98.2) | 78.7% (74.9–82.5) | 85.4% (80.8–89.2) | 0.70 |"),
    ("| classical | 38.3 ms | 26.1 | 5.5 ms | 175 MB |", BOARD, "| classical | 38.3 ms | 26.1 | 5.5 ms | 9.8 ms | 175 MB | 17 |"),
    ("**41.1 ms** | **24.3** | **8.9 ms** | 200 MB |", BOARD, "**41.1 ms** | **24.3** | **8.9 ms** | **16.3 ms** | 200 MB | 17 |"),
    ("48.1 ms a frame", BOARD, "**48.1 ms (20.8 a second)**"),
    ("from 76.4 to 61.4 ms", BOARD, "76.4 ms (13.1 a second)"),
    ("from 50 °C to 63 °C", BOARD, "| hottest zone | 49.9 °C | 61.9 °C | 62.8 °C |"),
    ("(1.1 W idle)", POWER, "| **1.1 W** | **2.9 W** | **1.7 W** | 18.3 |"),
    ("15–17 W (8–9 W idle)", POWER, "| 8–9 W | 15–17 W | ~7 W | 18.9 |"),
    ("0.0037 ms a crop", "results/laptop_npu/README.md", "**0.0037 ms**"),
    ("99.93%", "results/laptop_npu/README.md", "**99.93%**"),
    ("(0.19 ms)", BOARD, "| INT8, NPU, 1 crop, NHWC input | 0.193 ms |"),
    ("(0.12 ms)", BOARD, "| FP32, CPU, 4 threads | 0.117 ms |"),
    ("0.045 ms a crop", BOARD, "**0.045 ms**"),
    ("21.3–21.5 updates a second", LIVE, "| 3-3 | `20261027_105012` | 6 | 21.3 |"),
    ("27.6 updates a second", LIVE, "| 2-2, MJPG | `20261029_110745` | 6 | **27.6** |"),
    ("The 10 timed", LIVE, "| 3-3 | 5 | 33.85 | 33.40 | −0.45 |"),
    ("9.2 vs 8.7", "results/suggestions/REPORT.md", "| bench_0908 | 40.0 | after | 226 | 169 | 35 | 22 | 17% | 8.7 |"),
)
BANNED = ("INT4", "scored once", "held back from every", "every fall agrees", "same work",
          "fifth of the power", "15 minutes of the live", "zero falls", "kept that clean")


def read(path):
    with open(os.path.join(REPO, path), encoding = "utf-8") as f:
        return f.read().replace("\r\n", "\n")


def main():
    readme, bad = read("README.md"), 0
    for said, source, rests_on in CLAIMS:
        if said not in readme or rests_on not in read(source):
            print("DRIFT  %r  against %s: %s" % (said, source, "not in README" if said not in readme else "source changed"))
            bad += 1
    for word in BANNED:
        if word in readme:
            print("BANNED %r is back in the README" % word)
            bad += 1
    print("%d claims checked, %d wordings kept out, %d problems" % (len(CLAIMS), len(BANNED), bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
