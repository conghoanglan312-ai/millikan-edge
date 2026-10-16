# Detection accuracy on units never used in training

Hand-labelled frames (`edge/label.py`) inside the detection area; each quarter of a session and each unit weighted the same; 95% bootstrap intervals (2000 resamples). Match radius 8 px. Model rows filter the classical blobs at the labelled frames. Written by `edge/score_unseen.py`.

| unit | labelled frames | sure droplets in the area | blind review |
|---|---|---|---|
| unit3-2_1015 | 100 | 3359 | done |
| unit1-3_1015 | 100 | 2791 | done |

| units | row | precision | recall | F1 | false / frame | missed / frame |
|---|---|---|---|---|---|---|
| unit3-2_1015 | classical detector | 57.3% (52.6–62.4) | 86.5% (84.7–88.3) | 68.9% (65.5–72.4) | 21.65 | 4.54 |
| unit3-2_1015 | classical tracker | 76.6% (72.3–81.0) | 81.3% (78.6–83.7) | 78.9% (76.1–81.6) | 8.33 | 6.28 |
| unit3-2_1015 | FP32 classifier, first model — detector | 93.1% (88.7–96.6) | 86.1% (84.3–87.9) | 89.5% (87.2–91.4) | 2.16 | 4.66 |
| unit3-2_1015 | FP32 classifier, retrained with four more units — detector | 93.2% (88.9–96.7) | 85.5% (83.6–87.4) | 89.2% (86.9–91.2) | 2.10 | 4.88 |
| unit3-2_1015 | INT8 classifier, first model (RKNN simulator) — detector | 93.0% (88.7–96.6) | 86.1% (84.3–87.9) | 89.4% (87.2–91.4) | 2.17 | 4.67 |
| unit1-3_1015 | classical detector | 93.7% (89.9–96.7) | 88.9% (87.7–90.1) | 91.2% (89.3–92.9) | 1.68 | 3.09 |
| unit1-3_1015 | classical tracker | 94.4% (91.1–97.1) | 85.1% (83.6–86.6) | 89.5% (87.5–91.2) | 1.40 | 4.15 |
| unit1-3_1015 | FP32 classifier, first model — detector | 94.1% (90.3–97.1) | 88.9% (87.6–90.1) | 91.4% (89.5–93.1) | 1.56 | 3.10 |
| unit1-3_1015 | FP32 classifier, retrained with four more units — detector | 94.1% (90.4–97.1) | 88.5% (87.2–89.7) | 91.2% (89.2–92.9) | 1.55 | 3.21 |
| unit1-3_1015 | INT8 classifier, first model (RKNN simulator) — detector | 94.1% (90.3–97.1) | 88.9% (87.6–90.1) | 91.4% (89.5–93.1) | 1.56 | 3.10 |
| all unseen | classical detector | 69.8% (65.9–73.6) | 87.6% (86.5–88.7) | 77.7% (75.2–80.1) | 11.66 | 3.82 |
| all unseen | classical tracker | 84.0% (80.9–86.8) | 83.0% (81.4–84.5) | 83.5% (81.7–85.2) | 4.87 | 5.21 |
| all unseen | FP32 classifier, first model — detector | 93.5% (90.7–95.9) | 87.4% (86.3–88.5) | 90.3% (88.8–91.7) | 1.86 | 3.88 |
| all unseen | FP32 classifier, retrained with four more units — detector | 93.6% (90.7–96.0) | 86.8% (85.7–88.0) | 90.1% (88.5–91.5) | 1.83 | 4.04 |
| all unseen | INT8 classifier, first model (RKNN simulator) — detector | 93.5% (90.6–95.9) | 87.4% (86.2–88.5) | 90.3% (88.8–91.6) | 1.86 | 3.88 |

## The blind review, by where each spot came from

Spots inside the detection area. The reviewer saw every spot alike, in random order, with its source withheld.

| unit | spot | droplet (y) | not one (n) | can't tell (u) |
|---|---|---|---|---|
| unit3-2_1015 | a hand label some row missed | 645 | 6 | 8 |
| unit3-2_1015 | a detection with no label that a model kept | 57 | 112 | 11 |
| unit3-2_1015 | a detection with no label that no model kept | 1 | 1909 | 7 |
| unit1-3_1015 | a hand label some row missed | 411 | 2 | 48 |
| unit1-3_1015 | a detection with no label that a model kept | 71 | 75 | 5 |
| unit1-3_1015 | a detection with no label that no model kept | 0 | 11 | 0 |
