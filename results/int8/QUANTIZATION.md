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
