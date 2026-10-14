# INT8 quantization — what it costs (RKNN simulator, validation crops)

`results/fp32/model.onnx` → `model_int8.rknn` (77317 bytes), RKNN-Toolkit2 2.3.2, rk3588, w8a8, per-channel, calibrated on 499 training crops (half droplet). Written by `edge/convert.py`.

| metric | FP32 (onnxruntime) | INT8 (simulator) |
|---|---|---|
| ap | 0.9568 | 0.9582 |
| auc | 0.9977 | 0.9978 |
| pass rate, droplet | 0.9721 | 0.9738 |
| pass rate, phantom track | 0.0215 | 0.0220 |
| pass rate, sweep blob | 0.0121 | 0.0151 |
| pass rate, gate reject | 0.0084 | 0.0090 |

Same decision at 0.5 on 99.89% of 41038 crops; mean |Δp| 0.0019, max 0.2822.

## On the test set — bench_0908 hand labels (2026-10-14)

`edge/int8_test.py` filters the classical detector's blobs at the 150 labelled frames
by each model's decision (INT8 from the simulator), then `edge/score.py` scores them;
every row in `TEST_SCORE_bench_0908.md`. Detector stage, 95% intervals:

| | precision | recall | F1 | false / frame |
| :-- | :-- | :-- | :-- | :-- |
| classical, current code | 81.0% (72.0–88.3) | 95.8% (94.1–97.3) | 87.8% | 1.35 |
| FP32, offline filter | 92.0% (89.7–94.2) | 93.3% (91.3–95.4) | 92.6% | 0.49 |
| **INT8, offline filter** | **92.1% (89.8–94.3)** | **93.7% (91.7–95.6)** | **92.9%** | **0.48** |
| FP32, full pipeline replay (reference) | 92.6% (90.4–94.7) | 93.3% (91.3–95.4) | 93.0% | 0.45 |

INT8 and FP32 decide alike on 99.49% of the 1,568 test blobs; quantization costs
nothing measurable. The offline filter sits within 0.04 false detections a frame of
the full replay, because in the pipeline the tracker's state feeds back into which
faint blobs the detector keeps.
