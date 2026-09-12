# V_COT Project Goal

## Base Framework
Monet + Qwen2.5-VL-7B.

## Scientific Question
Can visual-evidence interventions identify and improve latent reasoning states that are genuinely grounded in task-relevant visual information?

## Core Hypothesis
Not every latent visual thought is equally useful or visually grounded. Positive and negative visual-evidence interventions can provide a supervision signal for latent-state credit assignment, allowing the model to strengthen useful visual latent states and suppress spurious ones.

## Target Method
Visual-Evidence-Gated Latent Reasoning.

The intended final method should preserve Monet's continuous latent visual reasoning paradigm while introducing evidence-aware latent supervision and, only after that is validated, evidence-aware on-policy latent optimization.

## Development Route
- V0: SFT-only latent contrastive supervision.
- V1: Evidence-gated latent supervision.
- V2: Integrate latent-level credit assignment into Monet VLPO.

## V0 Principle
Do not modify the model architecture or RL pipeline initially. The first version should only add:
1. Positive visual view.
2. Negative visual view.
3. Latent-state extraction.
4. Latent contrastive supervision.

## Success Criterion
Under the same evaluation protocol, the proposed method must outperform a successfully reproduced Monet baseline on the predefined aggregate benchmark score before the project advances to more complex stages.

Any claimed gain must be measured against the reproduced baseline under matched settings.

## Hard Constraints
- Do not drift into text-only Chain-of-Thought optimization.
- Do not add unrelated modules without experimental motivation.
- Do not change the core scientific question without explicit approval.
- Do not treat an unreproduced paper number as the working baseline.
- Do not move to V1 or V2 before V0 has produced evidence that the core hypothesis is useful.
- Previous conversations and remembered project context are secondary to this file. When there is a conflict, this file is the project authority unless explicitly revised.

## Decision Rule
Before any substantial code, loss, architecture, dataset, or training change, check whether the proposed change directly tests or advances the scientific question above. If not, do not implement it without first recording a new approved decision in `DECISIONS.md`.
