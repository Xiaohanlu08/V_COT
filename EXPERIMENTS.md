# Experiment Log

This file is the permanent record of experiments that were actually run. Planned experiments should stay in `CURRENT_STATE.md` until execution begins.

## Recording Rules
For every experiment, record:
- experiment ID,
- date,
- branch,
- commit SHA,
- base checkpoint,
- configuration file,
- dataset / benchmark,
- training settings,
- random seed,
- hardware,
- result metrics,
- comparison baseline,
- conclusion,
- next action.

A failed experiment must be recorded as carefully as a successful one. Do not delete failed runs from project history.

---

## EXP-0000 — Repository Initialization
**Status:** COMPLETED

**Purpose:** Establish project governance and reproducibility anchors before importing or modifying model code.

**Changes:**
- Added `PROJECT_GOAL.md`.
- Added `CURRENT_STATE.md`.
- Added `DECISIONS.md`.
- Added `EXPERIMENTS.md`.

**Model run:** None.

**Metrics:** None.

**Conclusion:** Repository control structure initialized. No scientific result has been produced yet.

**Next action:** Reproduce and freeze the official Monet baseline before starting V0.

---

## Experiment Template

```markdown
## EXP-XXXX — Short title
**Status:** RUNNING / COMPLETED / FAILED / ABORTED

**Date:** YYYY-MM-DD

**Branch:** `...`

**Commit:** `...`

**Tag:** `...` (if verified)

**Base checkpoint:** `...`

**Config:** `configs/...`

**Purpose:**
...

**Change from baseline:**
...

**Data / benchmark:**
...

**Training settings:**
- learning rate:
- epochs / steps:
- batch size:
- precision:
- seed:
- other:

**Hardware:**
...

**Results:**
| Metric | Baseline | This run | Delta |
|---|---:|---:|---:|
| ... | ... | ... | ... |

**Conclusion:** KEEP / REJECT / INCONCLUSIVE

**Reason:**
...

**Next action:**
...
```

## Reproducibility Rule
A result is not considered verified unless the exact commit, configuration, checkpoint, benchmark protocol, and numerical result are recorded here. Verified milestones should also receive a Git tag.
