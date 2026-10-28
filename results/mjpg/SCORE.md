# Detection accuracy, raw vs MJPG capture: unit 3-2, 2026-10-27

Unit 3-2 recorded twice back to back, labelled blind as A and B (assigned at random, revealed after the blind review): **A = MJPG** (Media Foundation, 30.0 fps), **B = raw YUY2** (DirectShow, 23.6 fps). Different sprays, so different droplets.

Hand-labelled frames (`edge/label.py`) inside the detection area; each quarter of a session and each unit weighted the same; 95% bootstrap intervals (2000 resamples). Match radius 8 px. Model rows filter the classical blobs at the labelled frames. Written by `edge/score_unseen.py`.

| unit | labelled frames | sure droplets in the area | blind review |
|---|---|---|---|
| unit3-2_1027_A | 100 | 870 | done |
| unit3-2_1027_B | 100 | 1247 | done |

| units | row | precision | recall | F1 | false / frame | missed / frame |
|---|---|---|---|---|---|---|
| unit3-2_1027_A | classical detector | 37.5% (29.9–46.6) | 79.2% (76.6–82.0) | 50.9% (43.6–58.5) | 11.47 | 1.81 |
| unit3-2_1027_A | classical tracker | 51.5% (43.9–61.0) | 57.6% (48.4–68.0) | 54.4% (48.0–61.1) | 4.71 | 3.69 |
| unit3-2_1027_A | FP32 classifier, first model — detector | 98.0% (97.0–99.1) | 74.4% (71.7–77.8) | 84.6% (82.7–86.8) | 0.13 | 2.23 |
| unit3-2_1027_A | FP32 classifier, retrained with four more units — detector | 98.3% (97.5–99.1) | 72.3% (68.9–76.2) | 83.3% (80.9–85.9) | 0.11 | 2.41 |
| unit3-2_1027_A | INT8 classifier, first model (RKNN simulator) — detector | 97.9% (96.8–99.0) | 75.3% (72.6–78.4) | 85.1% (83.3–87.2) | 0.14 | 2.15 |
| unit3-2_1027_B | classical detector | 84.6% (77.4–90.6) | 82.2% (78.1–86.3) | 83.4% (78.7–87.3) | 1.86 | 2.22 |
| unit3-2_1027_B | classical tracker | 92.8% (86.1–97.4) | 66.2% (57.2–75.5) | 77.2% (69.5–84.4) | 0.64 | 4.22 |
| unit3-2_1027_B | FP32 classifier, first model — detector | 93.5% (86.9–98.2) | 78.5% (74.8–82.2) | 85.4% (80.9–89.1) | 0.68 | 2.68 |
| unit3-2_1027_B | FP32 classifier, retrained with four more units — detector | 93.2% (86.1–98.3) | 77.5% (73.3–81.7) | 84.6% (79.7–88.8) | 0.70 | 2.81 |
| unit3-2_1027_B | INT8 classifier, first model (RKNN simulator) — detector | 93.3% (86.5–98.2) | 78.7% (74.9–82.5) | 85.4% (80.8–89.2) | 0.70 | 2.66 |

## The blind review, by where each spot came from

Spots inside the detection area. The reviewer saw every spot alike, in random order, with its source withheld.

| unit | spot | droplet (y) | not one (n) | can't tell (u) |
|---|---|---|---|---|
| unit3-2_1027_A | a hand label some row missed | 353 | 4 | 9 |
| unit3-2_1027_A | a detection with no label that a model kept | 35 | 7 | 11 |
| unit3-2_1027_A | a detection with no label that no model kept | 19 | 1085 | 38 |
| unit3-2_1027_B | a hand label some row missed | 425 | 2 | 7 |
| unit3-2_1027_B | a detection with no label that a model kept | 22 | 51 | 10 |
| unit3-2_1027_B | a detection with no label that no model kept | 23 | 112 | 18 |
