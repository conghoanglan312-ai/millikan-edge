# Results

Every number this fork publishes, with the file that produced it. Nothing here is
edited by hand except where a file says so.

## `baseline/` — the "before" column, frozen 2026-09-29 before any detector change

| file | what | made by |
| :-- | :-- | :-- |
| `live_check_2026-09-29.txt` | phantom rate on the worst-case fixture: **3.11 detections/frame against the 0.87 bar** (FAIL), with versions | the parent's `test/tools/live_check.py` |
| `eval_bench_0908.npz` | the frozen evaluation set: detection area (ROI, grid bands, static mask), session clock, and the 2026-09-11 classical predictions at the 150 hand-labelled frames | `edge/score.py --freeze` |
| `SCORE_bench_0908.md` | precision / recall / F1 on the hand labels, every row scored by the same code | `edge/score.py` |

`SCORE_bench_0908.md` reproduces the parent's HANDOVER §8.5 to the decimal (detector
81.8 / 95.6 / 88.2, tracker 95.8 / 85.4 / 90.3, rings 96.6 / 63.2 / 76.4, same
intervals), which is the check that the port of the scorer is faithful. The labels are
read from `cv/for_report/detection_accuracy/`; their hashes are printed in the report,
so a changed label file shows up as a changed hash.

## `latency/`

| file | what | made by |
| :-- | :-- | :-- |
| `detect_rk3588.json` | `detectDroplets()` per frame on the board's CPU: noisy fixture 4.62 ms mean (216 FPS), typical frames 5.70 ms (175 FPS) | `edge/bench_detect.py` on the RK3588S |
| `detect_vivobook.json` | the same frames on the laptop (x64 Python under emulation): 2.02 ms (496 FPS) and 2.92 ms (342 FPS). Identical detection counts on both machines | `edge/bench_detect.py` |
