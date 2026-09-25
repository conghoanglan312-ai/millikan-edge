# millikan-edge — handover

Everything a fresh session needs to start work here. Written 2026-09-25, the day
the parent project was submitted, by the Claude Code session that did most of
this month's CV work.

Read §6 before writing any code. It is the only decision that is still open, and
it changes what the README may claim.

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

| what | where (on the build machine) | why it matters |
| :-- | :-- | :-- |
| 126 recorded sessions, ~65 GB | `test/output/live_*/` | every frame this project has ever seen |
| fixtures | `test/fixtures/` | `bench_0908`, `one_droplet_slow`, the panel sweeps |
| the acceptance gate | `test/tools/check.py` | the parent's regression test |
| the labelling harness | `test/tools/detacc_label.py`, `detacc_score.py` | see §4 |
| the live-path check | `test/tools/live_check.py` | where the baseline number comes from |

Copy what you need in deliberately, and decide per item whether it belongs in
git. The 65 GB does not; a sampled, labelled subset does.

The labelled data itself **is** tracked, under
`cv/for_report/detection_accuracy/` — that came across with the clone.

---

## 3. The problem this fork should solve

Not a contrived portfolio exercise. **The classical detector produces phantoms,
and it is documented, measured, and unsolved.**

- `test/tools/live_check.py` fails its own acceptance bar: **3.11 detections a
  frame against a 0.87 baseline** on the worst-case fixture.
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

The lab is **not** needed. Confirmed tracks label themselves:

- **positives** — any track the tracker confirmed (`CONFIRM_POINTS = 6` sightings)
  and especially any that completed a legal fall; `falls.csv` in every session
  folder gives the column, the frame span and the outcome
- **negatives** — `REJECT_POINTS` records every blob the gates threw out, with
  the gate that rejected it

Estimated **~2 hours** to write the extractor, minutes to run it across 126
sessions, plus an hour if you want to eyeball a sample before trusting it.

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

The board purchase is not blocked by this; RK3588 is right either way.

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

- **Build on the x86_64 machine (RTX 4060) under WSL2 Ubuntu 22.04.** Not the
  ARM64 Vivobook: RKNN-Toolkit2 supports Linux x86_64/ARM64 and Windows x86_64,
  and Windows ARM64 is not on that list. The Vivobook runs classical CV fine
  (~28.7 fps measured) — it is the toolchain, not the compute.
- **Docker is optional.** Worth it only if host dependency conflicts appear; the
  conversion toolchain pins specific Python and numpy versions.
- **Expect the RKNN toolchain to be the schedule risk**, not the model. Thin
  documentation, rough edges.
- **Claude Code on a VPN machine:** a 403 on `/login` means the terminal is going
  out unproxied while the browser is on the VPN. Set `HTTP_PROXY`/`HTTPS_PROXY`
  to the proxy client's mixed port. For Clash Verge specifically, rules only
  apply in **Rule** mode — Global mode ignores them entirely — and
  `anthropic.com` / `claude.ai` need DIRECT rules or the session dies when the
  proxy comes up. A working extend script is at `test/clash_extend_script.js` in
  the parent.

---

## 9. Timeline and definition of done

| date | item |
| :-- | :-- |
| 28 Sep 2026 | order the RK3588 (Orange Pi 5 or Firefly ROC-RK3588) |
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
