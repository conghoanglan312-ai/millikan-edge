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

## `fp32/` — the patch classifier, trained 2026-10-09

| file | what | made by |
| :-- | :-- | :-- |
| `README.md`, `metrics.json`, `history.csv`, `model.onnx` | training run and model; validation-only numbers | `edge/train.py` |
| `DATASET.md` | the crops it was trained and validated on | `edge/dataset.py` |
| `PHANTOM_RATE.md`, `noisy_kept.jpg`, `noisy_dropped.jpg` | `live_check`'s fixture: 3.11 → 2.08 detections a frame; what was dropped (noise) and kept (droplets) | `edge/phantom_rate.py` |

## `int8/` — quantized for the RK3588 NPU

| file | what | made by |
| :-- | :-- | :-- |
| `model_int8.rknn`, `QUANTIZATION.md`, `quantization.json` | INT8 vs FP32 on validation crops, RKNN simulator: AP 0.9582 vs 0.9568, same decision on 99.89% | `edge/convert.py` |
| `TEST_SCORE_bench_0908.md` | INT8 on the test set: F1 92.9%, 0.48 false detections a frame (FP32 92.6% / 0.49, same filter) | `edge/int8_test.py`, `edge/score.py` |
| `model_int8_b8/16/32.rknn` | batched models for the NPU gate | `edge/convert_batched.py` |

## `fp32_v2/`, `int8_v2/` — retrained 2026-10-15 with four more apparatus units; not used on the board, it scored the same on both tests

| file | what | made by |
| :-- | :-- | :-- |
| `fp32_v2/metrics.json`, `history.csv`, `model.onnx` | trained on 146,026 crops (v1: 78,484), same validation: AP 0.9697 vs 0.9568, precision@0.5 85.1% vs 82.0%, droplet pass rate 95.1% vs 97.2% | `edge/train.py` |
| `fp32_v2/DATASET.md` | the crops: v1's recordings plus units 2-1, 3-4, 4-1, 4-2 | `edge/dataset.py` |
| `fp32_v2/TEST_SCORE_bench_0908.md` | every row on the test session; retrained FP32 92.2 / 93.3 / F1 92.7%, 0.48 false a frame — the same as v1 on this session | `edge/int8_test.py --tag _v2`, `edge/score.py` |
| `int8_v2/model_int8.rknn`, `QUANTIZATION.md`, `quantization.json` | INT8 vs FP32: AP 0.9713 vs 0.9697, same decision on 99.94% | `edge/convert.py --onnx results/fp32_v2/model.onnx` |

## `unseen/` — units never used in training (1-3, 3-2), scored 2026-10-16

| file | what | made by |
| :-- | :-- | :-- |
| `SCORE.md`, `score.json` | per unit and pooled (each unit and quarter weighted the same): classical detector 69.8 / 87.6 / F1 77.7%, 11.66 false a frame; first model INT8 93.5 / 87.4 / 90.3%, 1.86 (FP32 the same; retrained FP32 90.1%, 1.83); the blind review's answers by source | `edge/score_unseen.py`, INT8 from `edge/int8_test.py --fixtures` |

Truth: `labels/unit1-3_1015/`, `labels/unit3-2_1015/` (100 frames each, `edge/label.py`),
corrected by the blind review of `edge/score_unseen.py --todo`.

## `suggestions/` — what the hunt aid points the operator at, 2026-10-17

| file | what | made by |
| :-- | :-- | :-- |
| `REPORT.md`, `report.json` | every track the hunt aid ringed or auto-picked, replayed with and without the classifier (FP32) and judged from blind clips (477): wrong suggestions per 10 minutes 9.2 → 8.7 on the test session, 1.7 → 1.7 and 3.3 → 3.3 on the unseen units — no reduction; the measure was fixed in the script's header before any replay | `edge/suggestions.py` run / clips / review / report |

## `live/` — the board live on units held back for it (1-1, 3-3), 2026-10-27

| file | what | made by |
| :-- | :-- | :-- |
| `README.md` | 11 legal falls measured live (21.3–21.5 updates a second); each within 0.87 s of the operator's timing from a phone video of the screen, on average 0.53 s (1-1) and 0.35 s (3-3); one droplet's charge per unit | the board's sessions; the phone timings by hand |
| `falls_*.csv`, `charges_*.csv` | the board's own session output, unedited | `edge/live_viewer.py` → `edge/live_board.py` |
| `demo_fall_1-1.gif` | 10 s of a live fall, 2× speed | cut from the board's screen recording |

## `mjpg/` — the camera's MJPG mode (30 fps) against its raw mode, 2026-10-28

| file | what | made by |
| :-- | :-- | :-- |
| `SCORE.md`, `score.json` | unit 3-2 recorded raw and then MJPG, 100 frames of each hand-labelled blind (as A and B) with the blind second look: the classical detector falls from 84.6% to 37.5% precision on MJPG (1.86 → 11.47 false a frame); the INT8 classifier scores the two alike — F1 85.4% raw, 85.1% MJPG, 0.70 vs 0.14 false a frame, recall 78.7% vs 75.3% | `edge/score_unseen.py --role mjpg`, INT8 from `edge/int8_test.py --fixtures` |

Truth: `labels/unit3-2_1027_A/` (MJPG), `labels/unit3-2_1027_B/` (raw), from `edge/label.py`.

## `laptop_npu/` — the same classifier on a laptop NPU (Snapdragon X Elite), 2026-10-17

| file | what | made by |
| :-- | :-- | :-- |
| `README.md`, `laptop_npu.json` | INT8 (QDQ, same calibration crops) on the Hexagon NPU agrees with FP32 on 99.93% of the 41,038 validation crops; 0.058 ms for one crop, 0.0037 ms a crop in batches of 32 (RK3588 NPU: 0.193 and 0.045 ms) | `edge/laptop_npu.py` (native ARM64 Python, onnxruntime-qnn 2.7.0) |

## `board/` — measured on the RK3588S, 2026-10-11

| file | what | made by |
| :-- | :-- | :-- |
| `README.md` + JSON | INT8 on the real NPU agrees with FP32 on 99.90%; NPU per-crop cost 0.193 ms alone, 0.045 ms in batches of 32; whole pipeline: classical 38.3 ms/frame, classifier on the NPU (batched) 41.1 ms/frame, detector + gate 8.9 ms median | `edge/board_bench.py`, `edge/npu_batch.py` |
| `soak_npu_15min.json` | 15 minutes of the live path with its window, INT8 on the NPU: hottest zone 49.9 → 62.8 °C (throttling starts at 75 °C), every clock at its maximum throughout, 14.1 updates a second with no downward trend | `edge/soak.py` |
| `README.md`, last section | the live path with its window costs 62 ms a frame on the busiest recording; `edge/fast.py` (same answers, checked on 10,000 replayed frames) brings it to 55.8 ms; batching the track-to-blob matching (cost matrices bit-identical on every call) then 55.2 → 48.1 ms by frame timing, the dense spray 76.4 → 61.4 ms; the occasional stall is the latched-voltage read at each fall's end, ~230 ms | `edge/fast.py`, `fast.py exact` |
