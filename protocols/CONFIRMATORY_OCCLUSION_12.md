# Confirmatory Full-Image Occlusion Replication Protocol

## Scope
This protocol is frozen before running any new target-occlusion outcome on the confirmatory cohort. It tests whether Monet's recurrent latent trajectory is selectively more sensitive to occlusion of the benchmark-annotated task target than to matched same-size nuisance occlusions elsewhere in the same image.

## Frozen cohort
Source file:
`results/replication_cohort/vstar_direct_attributes_confirmatory12.json`

Expected SHA-256 of the canonical cohort JSON list:
`f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4`

Frozen dataset positions:
`[6, 11, 22, 23, 27, 53, 57, 59, 67, 71, 102, 112]`

Position 0 is excluded because it was used as the development sample for EXP-0006 through EXP-0008.

## Eligibility rule
All included samples were selected outcome-blind from the naturally-triggered VStarBench pool and satisfy:
- category = `direct_attributes`;
- official V*Bench annotation resolved;
- exactly one target object and one bbox;
- valid local image and bbox metadata;
- natural latent segment length = 10;
- position 0 excluded.

`relative_position` samples are not mixed into this cohort because relational questions require relation-aware interventions involving multiple objects.

## Counterfactual replay
For each sample:
1. Read the exact generated-token prefix preceding its natural `<abs_vis_token>` from the saved baseline JSONL.
2. Append that prefix directly as `prompt_token_ids`; no decoded-text retokenization is used for replay.
3. Keep the base textual prompt IDs identical across original, target-mask, and sham-mask conditions.
4. Force only the first subsequent generated token to `151666=<abs_vis_token>` for intervention runs.
5. Capture the 10 recurrent `st["pending"]` latent vectors, each of hidden size 3584.
6. For the original image, record the token that greedy decoding would have sampled before the one-time override. The sample is mechanically valid only if this pre-force token is already `151666`, confirming that the fixed prefix lands at the natural latent-entry boundary.

## Visual intervention
All conditions preserve the full source-image geometry.

Target intervention:
- neutral-mask exactly the official target bbox;
- fill RGB is the mean of a surrounding ring computed with the same operator used in EXP-0008.

Sham controls:
- 32 same-size boxes per sample;
- deterministic seed = `20260913 + dataset_position`;
- boxes must not overlap the target exclusion region formed by expanding the target bbox by one target width and one target height in each direction;
- each sham box uses its own surrounding-ring mean-RGB fill;
- sham boxes are sampled before inference and are independent of latent outcomes.

## Per-sample primary quantity
For a masked condition `m`:

`distance(m) = 1 - mean_t cosine(z_t(original), z_t(m))`

with `t = 1..10`.

Per-sample target specificity:

`specificity_i = target_distance_i - median(sham_distance_i)`

Positive specificity means the annotated task target perturbs the latent trajectory more than a typical matched nuisance occlusion.

Secondary within-image descriptors:
- target percentile among 32 sham distances;
- number of sham distances greater than or equal to the target distance;
- empirical one-sided within-image p = `(1 + count(sham >= target)) / 33`;
- target-to-median-sham distance ratio when the denominator is positive.

These within-image p-values are descriptive and are not the primary cross-sample inference.

## Cross-sample primary test
Primary confirmatory test:
- count samples with `specificity_i > 0`;
- exclude exact zeros from the sign-test denominator;
- use an exact one-sided sign test under `P(positive)=0.5`.

The primary hypothesis is supported only if:
1. cohort mechanical validity passes for all 12 samples;
2. median specificity across samples is positive; and
3. the exact one-sided sign-test p-value is `< 0.05`.

For `n=12` with no ties, this requires at least 10 positive specificity values.

Secondary aggregate test:
- exact sign-flip permutation test on the mean specificity over all 12 samples, enumerating all `2^12=4096` sign assignments.

The secondary result is reported but does not replace the primary sign-test decision rule.

## Failure / exclusion rules
No sample may be removed because its target-specificity result is weak, negative, or inconvenient.
A sample is marked mechanically invalid only for a pre-specified failure such as:
- frozen cohort SHA mismatch;
- missing image or annotation metadata;
- original pre-force next token is not `151666`;
- latent segment is not `(0,10)` after fixed replay;
- captured latent shape is not `10 x 3584` or contains non-finite values.

If any confirmatory sample is mechanically invalid, report it and do not silently replace it with another sample.

## Decision toward V0
A positive single-sample result is insufficient. V0 training remains blocked until this multi-sample confirmatory protocol supports target-specific latent sensitivity under the primary rule above. If the primary rule fails, the visual-evidence gate is considered unsupported for this intervention/operator and the method should be revised before training.
