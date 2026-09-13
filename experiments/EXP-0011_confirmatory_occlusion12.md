# EXP-0011 — Confirmatory 12-Sample Target-vs-Sham Occlusion

**Status:** COMPLETED; PRIMARY GATE PASSED  
**Protocol:** `protocols/CONFIRMATORY_OCCLUSION_12.md`  
**Frozen cohort SHA-256:** `f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4`

## Frozen positions
`[6, 11, 22, 23, 27, 53, 57, 59, 67, 71, 102, 112]`

## Mechanical validity
```text
mechanically_valid_samples: 12
mechanically_invalid_samples: 0
all_12_mechanically_valid: true
```

## Primary results
```text
positive specificity: 10/12
negative specificity: 2/12
specificity mean:   +0.0008347084124883016
specificity median: +0.0004564523696899414
target percentile mean:   83.85416666666667
target percentile median: 100.0
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p:      0.0009765625
primary_hypothesis_supported: true
```

Frozen primary rule:
```text
all 12 mechanically valid
AND median specificity > 0
AND exact one-sided sign-test p < 0.05
```
All conditions were satisfied.

## Per-sample specificity
```text
pos=  6  +0.0016880035400390625   pct=100.000
pos= 11  +0.0010933578014373780   pct=100.000
pos= 22  +0.0025942325592041016   pct=100.000
pos= 23  +0.0003161132335662842   pct=96.875
pos= 27  +0.0000140070915222168   pct=56.250
pos= 53  -0.0000067055225372314   pct=12.500
pos= 57  -0.0000065267086029053   pct=43.750
pos= 59  +0.0002937614917755127   pct=96.875
pos= 67  +0.0005125701427459717   pct=100.000
pos= 71  +0.0020875632762908936   pct=100.000
pos=102  +0.0004003345966339111   pct=100.000
pos=112  +0.0010297894477844238   pct=100.000
```

## Interpretation
Under the pre-specified direct-attribute/full-image neutral-occlusion protocol, masking benchmark-annotated task-relevant target evidence perturbs Monet's aligned recurrent latent trajectory more than matched same-size nuisance occlusions across samples. This supports target-specific visual sensitivity of the recurrent latent state under this protocol.

This does **not** prove answer-level utility or universal visual grounding. The earlier latent-start suppression experiment remained null after robust answer rescoring.

## Decision
The evidence gate for V0 SFT-only development is open. The confirmatory VStarBench cohort must remain evaluation/probing evidence and must not be used as V0 training data.
