# Detection accuracy on bench_0908 — the fork's gate

150 hand-labelled frames, after the blind review, inside the detection area; phases weighted by share of session time; 95% bootstrap intervals (2000 resamples). Match radius 8 px. Written by `edge/score.py`.

| row | precision | recall | F1 | false / frame | missed / frame |
|---|---|---|---|---|---|
| classical detector, 2026-09-11 code | 81.8% (72.8–88.9) | 95.6% (94.0–97.1) | 88.2% (82.7–92.3) | 1.28 | 0.26 |
| classical tracker, 2026-09-11 code | 95.8% (93.4–97.5) | 85.4% (80.9–89.8) | 90.3% (87.5–92.8) | 0.23 | 0.88 |
| rings shown to the operator, 2026-09-11 code | 96.6% (94.8–98.1) | 63.2% (56.6–70.3) | 76.4% (71.3–81.5) | 0.13 | 2.21 |
| classical, current code — detector | 81.0% (72.0–88.3) | 95.8% (94.1–97.3) | 87.8% (82.2–92.0) | 1.35 | 0.25 |
| classical, current code — tracker | 95.0% (92.5–97.0) | 85.5% (81.0–89.9) | 90.0% (87.2–92.6) | 0.27 | 0.87 |
| FP32 classifier after the gates — detector | 92.6% (90.4–94.7) | 93.3% (91.3–95.4) | 93.0% (91.3–94.6) | 0.45 | 0.40 |
| FP32 classifier after the gates — tracker | 97.2% (95.9–98.2) | 83.2% (78.2–88.2) | 89.6% (86.6–92.5) | 0.15 | 1.01 |
| FP32 classifier replacing area + circularity — detector | 90.0% (87.2–92.6) | 95.1% (93.5–96.6) | 92.4% (90.7–94.1) | 0.64 | 0.30 |
| FP32 classifier replacing area + circularity — tracker | 95.7% (94.1–97.0) | 84.2% (79.5–88.9) | 89.6% (86.5–92.4) | 0.23 | 0.95 |
| FP32 classifier after the gates, offline filter — detector | 92.0% (89.7–94.2) | 93.3% (91.3–95.4) | 92.6% (91.0–94.3) | 0.49 | 0.40 |
| INT8 classifier (RKNN simulator) after the gates, offline filter — detector | 92.1% (89.8–94.3) | 93.7% (91.7–95.6) | 92.9% (91.2–94.5) | 0.48 | 0.38 |

Inputs (sha256, first 16 hex):

- `cv/for_report/detection_accuracy/labels.json` a28e4cd8d0424b87
- `cv/for_report/detection_accuracy/review.json` d8803cdf76d87320
- `cv/for_report/detection_accuracy/sample.json` 2cca0df2b837edb8
- `results/baseline/eval_bench_0908.npz` f914ad4ed6b8a438
