# On the board — RK3588S (DC-AI588S), 2026-10-11

Ubuntu 22.04, kernel 6.1.118, NPU driver 0.9.8, RKNN runtime 2.3.2, Python 3.10,
ONNX Runtime 1.20.1, OpenCV 4.10; CPU governor `performance`; 34–52 °C during the runs.
Made by `edge/board_bench.py` and `edge/npu_batch.py`; raw JSON alongside.

## Accuracy on the device

INT8 on the real NPU vs FP32 on the board's CPU, all 41,038 validation crops
(`bench_accuracy.json`): **same decision on 99.90%**, mean |Δp| 0.0019.
Droplets through 97.3% vs 97.2%, phantom tracks 2.18% vs 2.15%.
Batched NPU calls vs one crop a call (`bench_batched.json`): same decision on 99.93%.

## The model alone, per call (`bench_model.json`, `bench_npu_batch.json`)

| | per call | per crop |
| :-- | --: | --: |
| FP32, CPU, 1 thread | 0.224 ms | 0.224 ms |
| FP32, CPU, 4 threads | 0.117 ms | 0.117 ms |
| INT8, NPU core 0, 1 crop, NCHW input | 0.313 ms | 0.313 ms |
| INT8, NPU, 1 crop, NHWC input | 0.193 ms | 0.193 ms |
| INT8, NPU, 8 crops | 0.513 ms | 0.064 ms |
| INT8, NPU, 16 crops | 0.823 ms | 0.051 ms |
| INT8, NPU, 32 crops | 1.448 ms | **0.045 ms** |

A 15k-parameter model is all call overhead on the NPU: one crop a call loses to the
CPU. Batched, the NPU is 2.6× cheaper per crop than the CPU's best. Handing the input
over as NHWC saves the runtime a conversion on every call (−38% at batch 1).

## The whole pipeline (`bench_pipeline*.json`)

`spray_labelled_0907`: 1,592 frames, ~26 candidate blobs a frame (dense spray).
Classifier after the gates, threshold 0.5. Per frame includes PNG decode from eMMC,
which a live camera does not pay.

| variant | per frame | FPS | detector + gate, median | p90 | peak RSS | falls |
| :-- | --: | --: | --: | --: | --: | --: |
| classical | 38.3 ms | 26.1 | 5.5 ms | 9.8 ms | 175 MB | 17 |
| classifier FP32, CPU, 1 crop a call | 44.4 ms | 22.5 | 11.3 ms | 26.0 ms | 193 MB | 17 |
| classifier FP32, CPU, 1 call a frame | 44.4 ms | 22.5 | 10.9 ms | 25.3 ms | 206 MB | 17 |
| classifier INT8, NPU, 1 crop a call | 47.7 ms | 21.0 | 13.7 ms | 32.6 ms | 188 MB | 17 |
| **classifier INT8, NPU, batched** | **41.1 ms** | **24.3** | **8.9 ms** | **16.3 ms** | 200 MB | 17 |

- Every variant measures the same 17 falls.
- The NPU batched gate adds 3.4 ms (median) to the classical detector, against
  5.4 ms on the CPU. Most of what is left is cutting the crops in Python; the NPU's
  own share is ~1.2 ms for 26 crops.
- At 8.9 ms (p90 16.3 ms) the detector with the classifier fits a 30 fps frame
  budget (33 ms) with room for the tracker.

## Heat: 15 minutes of the live session (`soak_npu_15min.json`), 2026-10-17

`edge/soak.py --minutes 15`: the live path with its window on the board's own
desktop (`--local`), classifier INT8 on the NPU (batched), on `spray_labelled_0907`
played round and round. Every thermal zone and the CPU and NPU clocks every 5 s.

| | start | after 15 min | peak |
| :-- | --: | --: | --: |
| hottest zone | 49.9 °C | 61.9 °C | 62.8 °C |
| big cores (4–5 / 6–7) | 2256 / 2304 MHz | same | never lower |
| NPU | 1000 MHz | same | never lower |

- **No throttling:** every clock stayed at its maximum throughout. The board's own
  first throttle point is 75 °C (trip point 0, passive); the warm-up levelled off at
  about 62 °C in the last five minutes.
- **Frame rate held:** 14.1 updates a second over the 15 minutes, per minute
  10.8–18.2, swinging with the 68 s loop and with no downward trend. That is the
  whole live path — the hunt aid's tests, rings and HUD drawn every frame — on the
  busiest recording; the 24.3 FPS above is the pipeline without its window.
