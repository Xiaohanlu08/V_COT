# EXP-0015 — Robust Visual_CoT Crop-Recoverability Audit

**Status:** COMPLETED / KEEP

**Date:** 2026-09-13

**Base framework:** Monet pinned at `08939998d3d643a73a316e349faa34f420429153`

**Purpose:** Quantitatively test whether the dominant `Visual_CoT` helper-image family is a scalable localized crop/zoom family suitable for same-sample evidence discrimination.

## Frozen cohort
- subset: `Visual_CoT`
- N = 64
- seed = `20260913`
- selection SHA-256 = `b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063`

## Frozen strong rule
A sample is strong recoverable iff:
```text
same-source NCC >= 0.90
AND
same-source NCC - deterministic wrong-source NCC >= 0.05
```

## Frozen family gate
```text
all 64 pairs numerically valid
AND strong fraction >= 0.90
AND median same-source NCC >= 0.95
AND median same-minus-wrong NCC margin >= 0.10
```

## Implementation note
The first Step-22 implementation was invalid and is not scientifically interpretable because float32 NCC denominator cancellation produced impossible `|NCC| > 1` values and a later `None` control caused a logging crash. Step 22b reused the exact same 64 already-extracted image pairs, retained all scientific thresholds, switched NCC accumulation to float64, rejected near-constant windows, enforced the mathematical NCC bound, and used a geometry-only deterministic fallback when the immediate cyclic wrong source could not fit the helper under any frozen scale.

## Verified result
```text
valid pairs: 64/64
helper smaller than source: 61/64 = 0.953125
strong recoverable: 61/64 = 0.953125
strong-fraction Wilson 95% CI: [0.8710034626, 0.9839309832]

same-source NCC:
  mean   0.9837053909
  median 0.9923840666
  p10    0.9653481874
  p25    0.9880255994
  min    0.7988491299
  max    0.9999968733

wrong-source NCC:
  mean   0.4803939568
  median 0.4728784714
  p90    0.6611139209
  max    0.9404166109

same-minus-wrong margin:
  mean   0.5033114340
  median 0.5103442276
  p10    0.3129825441
  min   -0.0171389966

crop_family_gate_passed: true
```

Non-strong rows:
```text
63121  helper 14x5,   same NCC 0.8551, margin -0.0171
76645  helper 221x11, same NCC 0.7988, margin +0.3753
107888 helper 24x11,  same NCC 0.9586, margin +0.0182
```
These rows remain recorded and were not silently removed from the audit.

## Conclusion
KEEP. The frozen family-level gate passed. This supports treating `Visual_CoT` as a localized crop/evidence family for the first V0 pilot. It does not yet validate a negative construction, loss function, or benchmark improvement.

## Next action
Run the pre-specified same-source geometry-matched sham validation in `protocols/V0_VISUAL_COT_SHAM_VALIDATION.md`. Only if that mechanical gate passes should the first V0 positive/negative data-pair operator be frozen.
