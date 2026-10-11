# Phantom rate — live_20260903_noisy

471 frames, median background, as `live_check.py` computes it. Bar: 0.87 a frame (2026-09-03). Classifier `results/fp32/model.onnx`, threshold 0.50. Written by `edge/phantom_rate.py`.

| detector | detections / frame | vs the 0.87 bar |
|---|---|---|
| classical (current code) | 3.11 | FAIL |
| classifier after the gates | 2.08 | FAIL |
| classifier replacing area + circularity gates | 2.12 | FAIL |

PASS/FAIL uses live_check's own tolerance: within 2x the bar.

## What the remaining detections are

The fixture is **not** droplet-free: the detections that survive the classifier
cluster in one grid column at changing heights, which is droplets falling (field
on, panel +0191 V). `live_check` counts every detection, droplets included, so its
0.87 bar mixes the two and cannot say how many phantoms are left.

Looked at directly (`noisy_dropped.jpg`, `noisy_kept.jpg`: 60 random crops each,
frame minus background, the classifier's probability in the corner): of the 1,465
blobs the classical detector keeps over the 471 frames, the classifier drops 484,
and those are almost all streaky analog noise and speckle; the 981 it keeps are
almost all round, compact droplets, with a handful of borderline noise around
0.54–0.62. So the classifier removes about a third of the detections, and the third
it removes is the phantoms. The hand-labelled `bench_0908` score is what puts a
number on both kinds.
