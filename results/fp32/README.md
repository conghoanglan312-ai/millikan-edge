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
