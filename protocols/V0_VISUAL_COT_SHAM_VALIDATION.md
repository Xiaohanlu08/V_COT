# V0 Visual_CoT Same-Source Sham Validation

## Status
Pre-outcome protocol for Step 23.

## Motivation
Step 22b passed the frozen Visual_CoT crop-family gate:
- 64/64 numerically valid;
- 61/64 strong recoverable;
- strong fraction 0.953125;
- median same-source NCC 0.992384;
- median same-minus-wrong margin 0.510344.

This licenses validation of a same-source matched-sham negative. It does not yet license training.

## Input eligibility
Use exactly the 61 Step-22b samples satisfying the already-frozen `strong_recoverable` rule. Do not redefine or add samples.

## Negative construction
For each eligible sample:
1. use the recovered positive source-space box from Step 22b;
2. define a negative box with exactly the same source-space width and height;
3. enumerate a 16 x 16 lattice of feasible top-left positions;
4. expand the positive box by 0.25 crop width/height on each side;
5. keep only candidate negative boxes with zero intersection with this expanded positive region;
6. require at least 8 valid sham positions;
7. select one valid position using a deterministic hash of `seed=20260913` and row index;
8. crop the same source image and resize the negative to the exact helper-image pixel dimensions using bicubic interpolation.

No image content, answer text, similarity score, or downstream model outcome is used to choose the negative.

## Frozen mechanical gate
Step 23 passes only if:
- every Step-22b strong sample has at least 8 valid sham positions;
- geometry violations = 0;
- pixel-identical selected negatives = 0.

If the gate fails, do not silently loosen the exclusion region or reduce the minimum sham count. Inspect the failures first.

## What passing means
Passing Step 23 freezes the first V0 Visual_CoT data-pair operator:

`source image -> official helper positive + same-source geometry-matched sham negative`

It does not validate a contrastive loss and does not imply benchmark improvement.
