# Detection accuracy on bench_0908 — the fork's gate

150 hand-labelled frames, after the blind review, inside the detection area; phases weighted by share of session time; 95% bootstrap intervals (2000 resamples). Match radius 8 px. Written by `edge/score.py`.

| row | precision | recall | F1 | false / frame | missed / frame |
|---|---|---|---|---|---|
| classical detector, 2026-09-11 code | 81.8% (72.8–88.9) | 95.6% (94.0–97.1) | 88.2% (82.7–92.3) | 1.28 | 0.26 |
| classical tracker, 2026-09-11 code | 95.8% (93.4–97.5) | 85.4% (80.9–89.8) | 90.3% (87.5–92.8) | 0.23 | 0.88 |
| rings shown to the operator, 2026-09-11 code | 96.6% (94.8–98.1) | 63.2% (56.6–70.3) | 76.4% (71.3–81.5) | 0.13 | 2.21 |
| classical detector, current code | 81.0% (72.0–88.3) | 95.8% (94.1–97.3) | 87.8% (82.2–92.0) | 1.35 | 0.25 |
| classical tracker, current code | 95.0% (92.5–97.0) | 85.5% (81.0–89.9) | 90.0% (87.2–92.6) | 0.27 | 0.87 |

Inputs (sha256, first 16 hex):

- `cv/for_report/detection_accuracy/labels.json` a28e4cd8d0424b87
- `cv/for_report/detection_accuracy/review.json` d8803cdf76d87320
- `cv/for_report/detection_accuracy/sample.json` 2cca0df2b837edb8
- `results/baseline/eval_bench_0908.npz` f914ad4ed6b8a438
