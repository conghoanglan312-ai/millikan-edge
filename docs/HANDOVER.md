# millikan-edge — handover

Everything a fresh session needs to start work here. Written 2026-09-25, the day
the parent project was submitted, by the Claude Code session that did most of
this month's CV work.

**Updated 2026-09-26:** board ordered (§8.1); the build machine is now the
Vivobook under WSL2, not the RTX 4060 laptop (§8).

**Updated 2026-09-29:** the board arrived and passed (§8.1); the "before" numbers
are frozen (§3); §4's plan for auto-labels was wrong about where the data is and is
rewritten; **§11 is where the work stands and the decisions waiting on the owner.**

Read §6 and §11 before writing any code.

---

## 1. What this repo is

A **personal fork** of a four-person competition project, seeded with its full
history. The parent is the 12th National Undergraduate Physics Experiment
Competition, Topic 6 (AI + physics experiment): a real-time assistant for the
**Millikan oil-drop experiment**, submitted 2026-09-25 and not expected to change
again.

This fork exists for one purpose: **an Edge AI portfolio piece**, taking the
droplet detector to an RK3588 board and recording what it costs.

```
CV droplet detection  →  ONNX  →  RKNN  →  INT8  →  RK3588
```

Then measure: latency, FPS, memory, accuracy delta before and after quantization.

**The deliverable is the numbers, not a demo.** The benchmark table is the
artefact; "it works" is a claim.

### Credit

The history carries two authors: 122 commits by the repo owner, 42 by a teammate
(`Sky Alva`) whose work is the app and LLM side (`app_js/`, `llm/`). The CV
module — `cv/pipeline.py`, `cv/api.py`, `cv/source.py`, the panel reader, the
live tutor's rules — is the owner's. `git log --author` settles any question.

Keep this repo **private until the competition result is known**, whatever the
parent's status.

---

## 2. What is NOT in this clone, and where it is

**`test/` is gitignored in the parent and therefore absent here.** That is where
almost everything this fork needs actually lives:

Parent clone on the Vivobook: `C:\Users\LAN\Documents\GitHub\Physics-Experiment-Assistant\`
(from WSL2: `/mnt/c/Users/LAN/Documents/GitHub/Physics-Experiment-Assistant/`).

| what | where (in the parent clone) | why it matters |
| :-- | :-- | :-- |
| 126 recorded sessions, ~65 GB | `test/output/live_*/` | every frame this project has ever seen |
| fixtures | `test/fixtures/` | `bench_0908`, `one_droplet_slow`, the panel sweeps |
| the acceptance gate | `test/tools/check.py` | the parent's regression test |
| the labelling harness | `test/tools/detacc_label.py`, `detacc_score.py` | see §4 |
| the live-path check | `test/tools/live_check.py` | where the baseline number comes from |

Copy what you need in deliberately, and decide per item whether it belongs in
git. The 65 GB does not; a sampled, labelled subset does. From WSL2, read the
65 GB in place through `/mnt/c/`; do not copy it into the Linux filesystem (C:
had 46 GB free on 2026-09-26).

The labelled data itself **is** tracked, under
`cv/for_report/detection_accuracy/` — that came across with the clone.

---

## 3. The problem this fork should solve

Not a contrived portfolio exercise. **The classical detector produces phantoms,
and it is documented, measured, and unsolved.**

- `test/tools/live_check.py` fails its own acceptance bar: **3.11 detections a
  frame against a 0.87 baseline** on the worst-case fixture.
  *Correction 2026-10-10:* that fixture contains real droplets (field on, falling
  in one column), and `live_check` counts every detection, so 3.11 vs 0.87 is not a
  pure phantom count. The classifier removes 484 of its 1,465 blobs (3.11 → 2.08 a
  frame), and by eye those 484 are the noise (`results/fp32/PHANTOM_RATE.md`). The
  hand-labelled score is the one that separates phantoms from droplets.
- The square lands on noise. The whole of 2026-09-21/22 went into guarding
  *around* it — a minimum track age before anything may be picked, an area
  tolerance for measuring tracks, a cap on how many re-acquisitions a fall may
  be stitched from.
- **Two hand-written detector gates were tried and both failed the acceptance
  gate outright**: one testing the blob before the `0` crossing (2 traverses
  against a baseline of 46, all five reference falls lost), one testing a second
  after it (4 of 5 reference falls lost). Both were reverted. The source note at
  the reverted site says the real fix is a shape or point-spread test, which is
  precisely a small learned classifier.

That is the strongest thing this portfolio piece has: **a baseline that a
competent hand-tuned classical gate could not fix, measured and written down
before any model existed.** Problem → what I did → measured result, with the
problem independently dated.

**Protect those numbers.** Copy `live_check`'s output and
`cv/for_report/detection_accuracy/` into the fork's own results directory before
touching the pipeline, so the "before" column cannot drift.

*Done 2026-09-29* — `results/baseline/`, indexed in `results/README.md`. `live_check`
re-run: still 3.11 against 0.87. The labels stay where they are and are pinned by
hash instead of copied; the evaluation region and predictions are frozen in
`eval_bench_0908.npz`. `edge/score.py` reproduces §8.5 of the parent's report to the
decimal, and scores the **current** code's detector at **81.0% precision, 95.8%
recall, 1.35 false detections a frame** — the row a classifier has to beat.

---

## 4. The labelled set — counted 2026-09-25

In `cv/for_report/detection_accuracy/labels.json`:

- **165 screens** — 150 distinct frames drawn at a fixed seed from four phases of
  one recording, plus 15 unannounced repeats that measure the labeller's own
  consistency
- **1,863 marks, of which 1,267 are "sure"** and 596 "unsure"
- each mark is `[x, y, sure]` in full-frame pixels

`sample.json`, `review.json` (a blind second look at the 459 spots where the
marks and the tracker disagreed) and `control_answers.json` / `control_key.json`
(40 empty spots and 40 real droplets, shown blind) are alongside it. The method
is in `cv/for_report/HANDOVER.md` §7.7 and §8.5.

**The catch: all of it is from one session (`bench_0908`).**

So: patch count is fine, session diversity is not. A model trained and tested on
one session's frames will report an accuracy that means nothing — the background,
focus and noise are shared.

### Getting the rest without going back to the lab

The lab is still **not** needed, but the plan written here on 2026-09-25 was wrong
about where the data is (checked 2026-09-29):

- **The 126 sessions cannot be used.** Each holds only `annotated_multi.mp4` — the
  ROI with the overlay burned in, so every tracked droplet carries a ring and a crop
  of it carries its own label — plus `falls.csv` (no frame spans) and `rejects.txt`
  (counts per gate, not positions). `REJECT_POINTS` lives one frame in memory and is
  never written out.
- **Raw frames exist only in `test/fixtures`**, dated from the frame files:

| fixture | captured | frames | droplets? | role |
| :-- | :-- | --: | :-- | :-- |
| `bench_0908` | 09-08 11:51 | 56,543 | yes | **test** — hand-labelled |
| `live_20260903_noisy` | 09-03 | 471 | yes | **eval** — `live_check`'s phantom rate; keep out of training |
| `one_droplet_slow` | 09-05 15:03 | 27,585 | yes | pool (saved pre-cropped; ROI from `roi.txt`, as `check.py` does) |
| `one_droplet_full` | 09-05 15:30 | 10,026 | a few | pool |
| `spray_labelled_0907` | 09-07 11:43 | 1,592 | yes | pool |
| `zoom_labelled_0907` | 09-07 12:28 | 5,368 | yes | pool |
| `voltage_sweep_0907` | 09-07 17:08 | 1,575 | **none** | pool |
| `panel_sweep_0908` | 09-08 17:33 | 15,896 | **none** | pool |
| `panel_sweep2_0908` | 09-08 21:25 | 12,771 | **none** | pool |
| `panel_sweep3_0909` | 09-09 17:04 | 37,367 | **none** | pool |
| `unit2-4_1009` | 10-09 16:51 | 24,277 | yes — sprays, field off | pool; unit 2-4. 77,568 confirmed crops, 67,819 on tracks ≥ 200 sightings (87% above the phantom level) — more than all earlier recordings together |

**Apparatus numbering (from 2026-10-09):** the lab has 16 units in 4 rows of 4. A unit
is `row-position`: row counted from the professor's seat (1–4), position counted left
to right as seen from the professor's seat (1–4). Record the unit with every new
recording; the fixtures above predate this and their units are unknown. 2026-10-09:
unit **2-4** (unit 2-3 showed no droplets when sprayed; dropped).

The sweeps are recordings of the voltage panel with nothing sprayed: every blob the
detector keeps in them is a phantom **with certainty**, which makes them the cleanest
negatives there are.

`edge/autolabel.py` replays a fixture through the unmodified pipeline (wrapped from
outside, as the parent's `detacc_replay.py` does) and writes every contour — kept,
or rejected and by which gate — with a 32×32 crop of frame and background, labelled
from the tracker afterwards. All ten fixtures but `live_20260903_noisy` have been
run; the output is in `data/autolabel/` (gitignored, ~230 MB), with contact sheets
in `data/autolabel/sheets/`.

**The tracker's "confirmed" is not a clean positive label**, measured two ways:

- against the hand labels on `bench_0908`: confirmed blobs are droplets **93.3%** of
  the time (85% on tracks of 6–50 sightings, 94.5% at 50+); kept-but-unconfirmed
  blobs are droplets **32%** of the time, so they are **not** safe negatives
- on `one_droplet_slow`, the noisy 09-05 session, by contrast against the sweeps'
  certain phantoms (peak over background noise in the crop): only **49%** of
  confirmed crops stand above the phantoms' 95th percentile — 19–32% on tracks
  under 200 sightings, 96% at 200+. The tracker there kept phantom tracks alive for
  up to 200 sightings, which is §3's problem again. `spray_labelled_0907`: 94%.

Use the hand-labelled `bench_0908` set as the **held-out test set** — it is
independent of the tracker, which the auto-labels are not.

---

## 5. What the detector currently does

`cv/pipeline.py`, `detectDroplets()`. There is no neural network anywhere in it.

1. **rolling background subtraction** — median over a 300 s window
   (`BG_WINDOW_S`), rebuilt when detections stay 4× above the session's own quiet
   rate
2. **adaptive threshold** — `max(Otsu, median + 6·MAD)`; the MAD floor exists
   because Otsu alone splits a histogram with no droplets in it
3. **shape gates** — area 30–900 px, circularity ≥ 0.52 (`CIRCULARITY_MIN`),
   plus masks for the grid lines, the panel margin and static marks
4. **track confirmation** — a blob must be seen 6 times before it is a track

A learned patch classifier replaces **step 3** and nothing else: given a ~24×24
crop around a candidate, is this a droplet or a speck. That is the smallest
change that addresses §3, and the easiest thing in the world to quantize.

Useful constants live near the top of `cv/pipeline.py` with the measurement that
set each one written above it. The comments in that file are the real
documentation — they record what was measured and what failed, not what the code
does.

---

## 6. THE OPEN DECISION

The original handover framed this as undecided. It is not, on the evidence in §3.

**Option A — train a small classifier.** Gives a real model to convert and
quantize, keeps the Edge AI story intact, and is motivated by a failure the
project already documented. The condition the original handover set — "only
justified if plain thresholding actually struggles" — **is met and measured.**

**Option B — benchmark the classical pipeline on the board.** Still a legitimate
edge story (ARM optimisation, throughput under constraint, CPU vs NPU), but it
skips the quantization and runtime-conversion steps that Edge AI job descriptions
name explicitly.

**Recommendation: A, with B's numbers kept as the baseline row.** The comparison
is the artefact — classical / FP32 / INT8 across latency, FPS, memory and
accuracy. Three rows beat one, and you get B for free on the way to A.

The board was ordered on 2026-09-26 without waiting on this (§8.1); it serves
either option.

---

## 7. Method cautions that will decide whether the numbers mean anything

1. **Split the test set by SESSION, not by frame.** Frames within a session share
   background, focus and noise. A random frame split will overstate accuracy
   badly, and it is the single most common way results like this become
   worthless.
2. **Keep the classical path in the repo and runnable.** It is the baseline row
   and the sanity check.
3. **Get the FP32 number and stop.** That is a publishable result even if the
   board never arrives, and it isolates "the model helps" from "the conversion
   preserved it".
4. **Each step should produce a number you could publish before starting the
   next.** The parent project's own lesson, learned the expensive way: define
   what done looks like before executing.
5. **Accuracy delta is measured against the hand-labelled set**, not against the
   tracker's own output. Otherwise you are measuring agreement with the thing you
   are replacing.

---

## 8. Environment

- **Build on the Vivobook under WSL2 Ubuntu** (ASUS Vivobook S15 S5507QA,
  Snapdragon X Elite, Windows 11 ARM64). This supersedes the 2026-09-25 plan to
  build on the RTX 4060 machine. RKNN-Toolkit2 ships Linux ARM64 packages as well
  as x86_64 (`rknn-toolkit2/packages/arm64` in airockchip/rknn-toolkit2, checked
  2026-09-26); only native Windows on ARM is unsupported, and WSL2 is Linux. The
  Vivobook runs classical CV fine (~28.7 fps measured).
  - No CUDA on the Vivobook: training runs on the CPU. Slower, and fine for a
    ~24×24 patch classifier. The RTX 4060 machine stays available for faster
    training and for x86-only tools (reflashing, §8.1).
- **WSL2 installed 2026-09-30:** Ubuntu 22.04, aarch64, Python 3.10.12 — the
  board's Python. No reboot was needed. Default user `lan` (passwordless sudo; set
  one with `passwd`). Toolchain in `~/venvs/rknn`, installed by `edge/setup_wsl.sh`
  in ~4 min: RKNN-Toolkit2 **2.3.2** (matches the board's runtime), torch 2.2.0,
  onnx 1.16.1, onnxoptimizer 0.3.8 (no ARM64 wheel — built from source, hence
  `build-essential cmake`), numpy 1.26.4. `pip freeze` in `~/rknn-pkg/freeze.txt`.
  From Git Bash, call it with `MSYS_NO_PATHCONV=1 wsl.exe ...`, or `/mnt/c/...`
  paths get rewritten into Windows ones.
- **The code already runs on Linux.** The Windows-only parts of `cv/` — the
  camera backend choice in `cv/source.py`, the DPI call in `cv/pipeline.py`, the
  `winsound` beep — sit behind `sys.platform` checks or a `try`. Two exceptions:
  `cv/UVC_check.py` hard-codes DirectShow and needs a Linux path before it is used
  on the board, and the 30 fps Media Foundation result in `cv/source.py` is a
  Windows measurement — on the board OpenCV uses V4L2, so capture fps must be
  measured again (it belongs in the benchmark anyway). `app_js/wrapper.py` is
  pywin32-only and out of scope.
- **Docker:** not needed to start. Recommended after the FP32 result: put the
  conversion toolchain in a Dockerfile so its pinned Python and numpy versions
  travel with the repo — that is most of the "reproducible" criterion in §9.
- **Expect the RKNN toolchain to be the schedule risk**, not the model. Thin
  documentation, rough edges.
- **Claude Code on a VPN machine:** a 403 on `/login` means the terminal is going
  out unproxied while the browser is on the VPN. Set `HTTP_PROXY`/`HTTPS_PROXY`
  to the proxy client's mixed port. For Clash Verge specifically, rules only
  apply in **Rule** mode — Global mode ignores them entirely — and
  `anthropic.com` / `claude.ai` need DIRECT rules or the session dies when the
  proxy comes up. A working extend script is at `test/clash_extend_script.js` in
  the parent.

### 8.1 The board — ordered 2026-09-26, arrived and passed 2026-09-29

**Dingchang (定昌) DC-AI588S V02: RK3588S, 8 GB RAM, 32 GB eMMC**, with a 12 V
adapter.

| | |
| :-- | :-- |
| NPU | 6 TOPS, INT4/INT8/INT16. Identical on every RK3588/RK3588S; the listing's "AI" tag names the board line, not a better chip |
| CPU | 4× Cortex-A76 + 4× Cortex-A55, up to 2.4 GHz |
| OS | Ubuntu 22.04.5 LTS, installed by the vendor (the spec sheet lists only Android 12) |
| camera | 1× USB 3.0 + 1× USB 2.0 host (MIPI CSI also present) |
| power | 12 V DC, 5.5×2.5 mm barrel, 3–5 A. The USB-C port is OTG (debug, flashing), not power. All USB ports share 3 A |
| storage | 32 GB eMMC; M.2 2242 PCIe SSD slot; TF card up to 128 GB |
| display, network | 1× HDMI (the product photo shows two; the spec sheet says one); gigabit Ethernet |
| spec sheet | [`docs/DC-AI588S-V02_spec.pdf`](DC-AI588S-V02_spec.pdf) — vendor 产品规格书 V1.0, 2026-01-04 |

**The listing excludes 7-day no-reason returns.** A return needs the board to
differ from the order (e.g. no Ubuntu), so run the checks below the day it
arrives. Support for images and the SDK is the vendor's after-sales engineers.

**Arrival checks** (~1 h, on the board):

| check | how | if it fails |
| :-- | :-- | :-- |
| Ubuntu, and which | `lsb_release -a` | Android → return. 20.04 and 22.04 are fine. 18.04's Python 3.6 is too old for RKNN-Toolkit-Lite2 (needs ≥ 3.7): workable, but ask for a newer image |
| NPU driver | `dmesg \| grep -i rknpu` prints a version | Nothing → ask support for an image with the driver; return if none comes within a few days. An old version is not a fault: pin the RKNN runtime, and the host toolkit, to a matching version |
| camera | plug into USB 3.0, `ls /dev/video*`, grab one frame with OpenCV | Try the USB 2.0 port; `v4l2-ctl --list-devices` |
| NPU runs | `pip install rknn-toolkit-lite2`, run its bundled ResNet example | Almost always a runtime/driver version mismatch → match the versions, or ask support |

Return only if it is not Ubuntu or the NPU cannot be used at all; everything else
is fixable. Once it passes, write the Ubuntu version, `uname -r`, the NPU driver
version and the RKNN runtime version into the repo — they are part of
"reproducible".

**Results, 2026-09-29 — passed; nothing to return:**

| | |
| :-- | :-- |
| OS | Ubuntu 22.04.5 LTS, kernel `6.1.118` aarch64, GNOME desktop, user `ztl`; Docker installed |
| NPU driver | **0.9.8** (`/sys/kernel/debug/rknpu/version`). The `can't request region` lines in `dmesg` are harmless: the NPU runs |
| RKNN runtime | `librknnrt` **2.3.2**, preinstalled — so the host RKNN-Toolkit2 must be **2.3.2** too |
| NPU runs (C) | Rockchip's `rknn_common_test`, MobileNetV1 INT8: **2.95 ms/inference (~338 FPS)**, dog image → dog class |
| NPU runs (Python) | RKNN-Toolkit-Lite2 2.3.2, bundled ResNet18: "space shuttle" at 0.9997 |
| Python | 3.10.12; installed offline: numpy 1.26.4, opencv-python-headless 4.10.0.84, pandas 2.2.3, scipy 1.15.3, rknn-toolkit-lite2 2.3.2 (+ psutil, ruamel.yaml). No pip came with the image; pip 26.2.1 was bootstrapped from its own wheel |
| RAM, disk | 7.7 GiB; 29 GB root, 21 GB free |
| CPU | 8 cores, governor `performance` |
| camera | **The lab's capture dongle works (2026-10-09):** MacroSilicon `345f:2109`, UVC 1.00 → `/dev/video0`, no driver needed. 1280×720: **MJPG 30.0 fps, YUYV 22.9 fps** (measured 10 s each, OpenCV V4L2). `autoROI` finds the reticle at x21 y36 909×645, 61.0 px/div — the same framing as `bench_0908`. Used in the **USB 2.0** port: the dongle is too thick for the USB 3.0 port next to the USB-C cable. MJPG is compressed and every recorded fixture is uncompressed, so which one the live path uses is a decision, not a default |

**Access without a network: ADB over the USB-C port.** Board USB-C → laptop USB-A.
It enumerates as `2207:0006` "rk3xxx"; Windows binds its in-box WinUSB driver by
itself, and `adb shell` is a root shell with no password. `adb push` moves files at
~33 MB/s. Google's platform-tools (x64) run fine on the Vivobook. The board has no
Wi-Fi, and no Ethernet cable was needed for any of the above: packages are
downloaded on the laptop (`pip download --platform manylinux2014_aarch64
--python-version 310`) and pushed. Staging lives in `/root/edge_setup/` and
`/root/millikan-edge/` on the board.

**No RTC battery:** the clock comes up wrong after a power loss. Set it before
anything that timestamps: `adb shell "date -s @$(date +%s)"`.

**Reset, 2026-09-29 ~23:12 — a loose adapter plug.** The board lost power twice in a
row and a file it had just written came back empty. The owner found the adapter
loose; not a board fault. Keep the habit it taught: `sync` after writing results on
the board.

**Reflashing, only if ever needed:**

1. **USB stick** — the spec sheet lists PC/U盘 upgrade; get the image and steps
   from vendor support. Needs no driver on any laptop.
2. **`rkdeveloptool`** (open source) on Linux — from WSL2 through usbipd-win,
   which has an ARM64 build.
3. **Rockchip RKDevTool on x86 Windows** — the RTX 4060 machine, not the
   Vivobook: Windows on ARM cannot load x86/x64 kernel drivers.

---

## 9. Timeline and definition of done

| date | item |
| :-- | :-- |
| 26 Sep 2026 | ✓ board ordered: DC-AI588S, RK3588S 8+32G (§8.1) |
| 29 Sep 2026 | ✓ board arrived; arrival checks passed except the camera (§8.1) |
| 29 Sep 2026 | ✓ "before" numbers frozen; auto-labels extracted from every raw fixture (§3, §4, §11) |
| 30 Sep 2026 | ✓ WSL2 and the RKNN toolchain installed on the Vivobook (§8) |
| Oct 2026 – Jan 2027 | build window |
| **Jan 2027** | **MUST: inference running on the board, numbers recorded** |
| Nov–Dec 2026 | BEST: README with the before/after table, demo video |
| Dec 2026 | BEST: Gitee mirror |
| Mar–May 2027 | internship applications — what this exists to support |

**Done means:**

- [ ] model (or classical pipeline) running inference on the RK3588
- [ ] before/after recorded: latency, FPS, memory, accuracy delta
- [ ] reproducible — someone else could follow the repo and get the same numbers

Everything past that is upside.

Commit as you build. A repo that appears fully formed three months before
applications reads as job-hunt preparation, and the activity period is the first
thing an interviewer checks after opening the link.

---

## 10. Inherited conventions worth keeping

From the parent, because they earned their place:

- **Never invent a threshold.** Every constant in `cv/pipeline.py` has the
  measurement that set it written above it. Keep that habit; it is most of why
  the baseline in §3 is defensible.
- **An acceptance gate before a refactor.** `check.py fixtures` must keep five
  reference falls and stay above a traverse baseline. Build the equivalent here
  before changing the detector: a fixed labelled set and a score, run before and
  after every change.
- **The operator's observation is evidence.** When the bench and the logs
  disagree, the bench is right and the logs need explaining.

---

## 11. Where the work stands — 2026-09-29

### Built

| path | what |
| :-- | :-- |
| `edge/common.py` | paths, the fixture registry (dates, roles), the parent's matching rule |
| `edge/score.py` | **the fork's gate** (§10's "fixed labelled set and a score"): any row of predictions against the hand labels, all rows by the same code |
| `edge/autolabel.py` | replay → every contour, 32×32 crops, tracker labels. **Validated:** with its hooks in place `one_droplet_slow` gives the same 52 traverses and the same 14 kinematically legal falls as the parent's unhooked `check.py` on the same code, which passes |
| `edge/contact_sheet.py` | 96-crop sheets per class, for eyeballing labels |
| `edge/bench_detect.py` | `detectDroplets()` latency on fixed frames, any machine |
| `results/` | the frozen baseline, scores and latencies — indexed in `results/README.md` |

### Numbers so far

| | |
| :-- | :-- |
| classical detector, current code, `bench_0908` | precision 81.0%, recall 95.8%, F1 87.8%, **1.35 false detections a frame** |
| phantom rate, `live_check` | **3.11** detections a frame against the 0.87 bar |
| `detectDroplets()` latency | board CPU 4.62 ms (noisy) / 5.70 ms (typical); Vivobook 2.02 / 2.92 ms; identical detection counts on both |
| NPU sanity | MobileNetV1 INT8 2.95 ms on the board |

### Labels available (every raw fixture but the eval one)

| | count, excluding `bench_0908` |
| :-- | --: |
| confirmed-track crops, any length | 17,936 |
| …on tracks of ≥ 50 sightings | 13,044 |
| …on tracks of ≥ 200 sightings | 5,519 |
| kept blobs in the droplet-free sweeps — certain phantoms | 2,809 |
| gate-rejected crops (sampled, ≤ 40 a frame) | 226,753 |

How clean "confirmed" is depends on the session, measured as the share of crops
whose peak stands above the sweeps' phantom 95th percentile (a proxy — a defocused
real droplet fails it too): `spray_labelled_0907` 94%, `one_droplet_full` 85%,
`one_droplet_slow` 49%, `zoom_labelled_0907` 21%; at ≥ 200 sightings every session
is 85–98%. On `bench_0908`, where there is truth, confirmed blobs are 93.3% droplets.

`one_droplet_slow` also shows the phantom problem at the level of results: the
current code finds 14 kinematically legal falls where the operator timed 5 (the
five are all kept; the nine extra are unverified — other droplets or phantom
tracks).

### Decisions waiting on the owner

1. ~~**§6, option A**~~ — **decided 2026-09-30: A**, train the classifier.
2. ~~Positives, negatives, split~~ — **decided 2026-10-09**, on the owner's blind
   review of 236 tracks (`data/review/REPORT.md`; droplets per 20 shown, by track
   length 6–49 / 50–199 / 200+): spray 19 / 18 / 20, unit 2-4 11 / 11 / 17,
   one_droplet_slow 0 / 2 / 19, zoom 0 / 0 / 9 of 16.
   - **Positives:** spray, every confirmed track; unit 2-4, one_droplet_slow and
     one_droplet_full, tracks ≥ 200 sightings; zoom, none.
   - **Negatives:** the phantom tracks the review found (zoom confirmed < 200,
     one_droplet_slow confirmed < 50), kept blobs in the droplet-free sweeps, and
     gate rejects.
   - **Split by day:** train 10-09 (unit 2-4), 09-05, 09-09 — 71,659 positive /
     201,974 negative crops; validate 09-07, 09-08 evening — 2,143 / 88,588; test
     `bench_0908` (09-08 morning, hand labels) only.
   - Unit 2-4 is 95% of the positives: cap crops per track and balance recordings.

### Ready to train (2026-10-09)

- `edge/dataset.py` → `data/dataset/` (gitignored): train 78,484 crops (32,468
  droplet / 46,016 noise), validate 41,038 (1,831 / 39,207); per-recording table in
  `data/dataset/SUMMARY.md`.
- `edge/train.py`, in WSL: 15,345-parameter conv net on the 24×24 frame-minus-
  background centre; class- and recording-balanced sampling; the best epoch is
  chosen on validation AP only. Smoke run passed, ONNX exported, and RKNN-Toolkit2
  2.3.2 converted that ONNX for rk3588 (FP16; it logs `E RKNN: Unkown op target: 0`
  yet returns 0 and writes the .rknn — check on the board).
- Full run: `wsl -d Ubuntu-22.04 -- bash -c "~/venvs/rknn/bin/python
  /mnt/c/Users/LAN/millikan-edge/edge/train.py"` — estimated ~1 h on the CPU.
- **Trained 2026-10-09** (`results/fp32/`): 40 epochs, ~58 s each. Best epoch 26 on
  validation: AP 0.957, AUC 0.998; at 0.5 it passes 97.2% of droplets, 2.2% of the
  review's phantom tracks, 1.2% of sweep blobs, 0.8% of gate rejects. Crop-level and
  validation-only — next is the classifier in the pipeline, scored on the test set.

### First before/after (2026-10-10) — `results/fp32/README.md`

- Classifier in the pipeline (`edge/gate.py`, from outside `cv/`), threshold 0.5.
  **Detector stage on the test set: false detections 1.35 → 0.45 a frame,
  precision 81.0 → 92.6%, recall 95.8 → 93.3%, F1 87.8 → 93.0%.** All 19 legal
  falls kept, durations within 0.04 s. Tracker stage about unchanged. Kept variant:
  classifier *after* the gates ("replacing" them lost a legal fall).
- `live_check` fixture: 3.11 → 2.08 a frame; the removed third is the noise (§3).
- INT8 (`results/int8/`): same decision as FP32 on 99.89% of validation crops,
  AP 0.9582 vs 0.9568 — on the RKNN simulator. Next: the NPU on the board.
- Replays of `bench_0908` run on Windows: from WSL, reading the 21 GB of PNGs across
  `/mnt/c` was twelve times slower.

### On the board (2026-10-11) — `results/board/README.md`

- INT8 on the real NPU = FP32 on 99.90% of validation crops.
- One crop per NPU call loses to the CPU (0.31 vs 0.12 ms): a 15k-parameter model is
  all call overhead. Batched (one model per batch size 8/16/32, NHWC input) it is
  0.045 ms a crop. `rknn_batch_size` gave wrong answers; the batch is baked into the
  ONNX export instead (`edge/convert_batched.py`).
- Whole pipeline, dense spray (26 blobs a frame): classical 38.3 ms/frame, 175 MB;
  classifier INT8 on the NPU, batched, 41.1 ms/frame, 200 MB; detector + gate 8.9 ms
  median (p90 16.3) — inside a 30 fps budget. Same 17 falls in every variant.

### Live on the board, window on the laptop (2026-10-11)

`python edge/live_viewer.py [--record]` — the whole live session from the laptop:
the board runs the pipeline (classifier INT8 on the NPU, batched, after the gates)
with its window streamed over the USB-C cable, and clicks and `q` go back. Click a
droplet to pick it, as in the parent's window. `--record` saves what you see as MP4
in `data/live/`; the board's session folder (falls.csv, report) is copied there too.
`--source frames:/root/test/fixtures/spray_labelled_0907` rehearses on a recording.

- Board side `edge/live_board.py`: takes over the pipeline's window calls from
  outside `cv/` (imshow → stream, setMouseCallback → laptop clicks, waitKey → laptop
  keys). The parent's `--publish` stream was sent before the HUD, and its remote
  clicks needed a real window.
- Chinese HUD text (平衡/提升) is drawn with Noto Sans CJK on the board: `cv2.putText`
  draws `?` for it — in the parent's own window too.
- Rehearsed 2026-10-11 on `spray_labelled_0907`: ~11 frames a second in the viewer,
  a laptop click picked the droplet under it, clean stop on `q`.
- 2026-10-12, owner's hands-on test: clicks, timer, HUD all fine, but choppy — 7.9
  updates a second, 31% of gaps over 150 ms (measured from their screen recording).
  Cause: the pipeline shows every 2nd frame (`SHOW_EVERY`, set for the parent's
  14 ms Windows imshow). `live_board.py` sets `P.SHOW_EVERY = 1`: 21.8 updates a
  second, gap median 44 ms, p90 59 ms on the same recording. The viewer prints this
  smoothness line at the end of every session.
- At the bench: dongle → board **USB 2.0** port (the USB 3.0 port fouls the USB-C
  plug), board USB-C → laptop USB-A, 12 V seated. `adb` lives in
  `data/tools/platform-tools/` (gitignored).

### Next, needing only minutes of the owner

- ~~WSL2 and the toolchain~~ — done 2026-09-30 (§8).
- **Track review** (~9 min): `python edge/review_tracks.py review` — 176 blind clips
  of confirmed tracks from the three droplet recordings, y / n / u per clip; then
  `report`. It settles decisions 2 and 4 with a person's eye instead of a proxy.
- **Live, not only replayed (2026-09-30).** The fork's claim is real-time on an edge
  device, so replays alone undersell it. The plan:
  - **Recorded fixtures stay the source of every published number** — they are what
    makes the table reproducible by someone else.
  - **No webcam.** The live path accepts a camera only if its picture shows the
    reticle (`LiveSource.accept` = `autoROI`), so a webcam cannot run the pipeline
    live; it could only time capture.
  - **A short lab trip after the owner's vacation, for the dongle.** The apparatus's BNC composite output
    (视频输出) reaches the laptop through a converter that Windows recognised with
    no driver installed — which on Windows means USB video class, which the board's
    kernel has built in (§8.1). Very likely fine, never tried; the dongle's model is
    not recorded anywhere. Take the board, its 12 V adapter, the laptop and the USB-C
    cable; dongle into the board's USB 3.0 port; over ADB: `v4l2-ctl --list-devices`,
    `--list-formats-ext`, real FPS per format (MJPG vs YUYV), `autoROI` accepts the
    frame, a short live run. ~30 min. Optional, same trip: record raw sessions with
    droplets (`test/tools/raw_capture.py`) — positives currently come from only two
    days (§11).
  - **One lab session at the end**, once INT8 runs on the board: the board live on
    the apparatus, a few falls against a stopwatch, the demo video. That is the
    headline, and the tie to the parent's real-time claim.
