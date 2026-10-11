# FP32 classifier — run `fp32_20261009_2255`

Trained by `edge/train.py` in WSL (torch 2.2.0, CPU), 40 epochs × 60,000 sampled crops,
~58 s an epoch. The epoch kept is the one with the best **validation** average
precision; the test set (bench_0908 hand labels) has not been used.

**Validation (09-07, 09-08 evening), best epoch 26:** average precision 0.957, ROC AUC
0.998. At a 0.5 threshold, the share of each kind the classifier lets through:

| kind | crops | passes |
| :-- | --: | --: |
| droplet | 1,831 | 97.2% |
| phantom track (confirmed by the tracker, rejected by the owner's review) | 4,362 | 2.2% |
| blob kept in a droplet-free sweep | 992 | 1.2% |
| area / circularity gate reject | 33,853 | 0.8% |

These are crop-level numbers on validation days. The result that counts comes next:
the classifier in place of the area and circularity gates, scored by `edge/score.py`
on the hand labels and by `live_check.py` on the phantom fixture.

Files: `model.onnx` (15,345 parameters, input 1×1×24×24), `metrics.json`,
`history.csv`, `DATASET.md` (the crops it was trained and validated on).

## The test — bench_0908 hand labels, scored once (2026-10-10)

The classifier in the pipeline, threshold 0.5, both variants fixed before scoring;
`TEST_SCORE_bench_0908.md` has every row with 95% intervals. Same environment for
all rows (Windows, OpenCV 5.0).

| detector stage | precision | recall | F1 | false / frame | missed / frame |
| :-- | :-- | :-- | :-- | :-- | :-- |
| classical, current code | 81.0% (72.0–88.3) | 95.8% (94.1–97.3) | 87.8% | 1.35 | 0.25 |
| **classifier after the gates** | **92.6% (90.4–94.7)** | 93.3% (91.3–95.4) | **93.0%** | **0.45** | 0.40 |
| classifier replacing area + circularity | 90.0% (87.2–92.6) | 95.1% (93.5–96.6) | 92.4% | 0.64 | 0.30 |

- **False detections a frame fall by two-thirds** (1.35 → 0.45) for 2.5 points of
  recall; the precision intervals do not overlap.
- **Measurements are unchanged.** "After the gates" keeps all 19 legal falls of the
  session, each within 0.04 s of the classical duration, from the same 165
  traverses. "Replacing" loses one (38.86 s) and adds 5 traverses.
- **The tracker stage barely moves** (F1 90.0 → 89.6; precision 95.0 → 97.2, recall
  85.5 → 83.2): the tracker was already filtering most phantoms by persistence. What
  the classifier changes is how much noise reaches it — 0.45 instead of 1.35 false
  blobs a frame to associate, coast and discard.

Kept variant for INT8 and the board: **after the gates**.
