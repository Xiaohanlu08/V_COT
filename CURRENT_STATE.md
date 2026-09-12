# Current State

## Project Stage
Repository initialization / pre-baseline.

## Current Objective
Establish a reproducible Monet baseline before implementing any new latent-supervision method.

## Current Branch
`main`

## Last Verified State
The repository has been initialized with the project anchor files. No Monet code, environment, baseline result, or modified method has yet been verified in this repository.

## Current Task
Prepare the repository and execution environment for the official Monet baseline.

## Next Milestones
- [ ] Import or clone the exact Monet implementation to be used as the baseline.
- [ ] Record the upstream repository URL and upstream commit SHA.
- [ ] Reproduce official inference on at least one provided example.
- [ ] Reproduce the selected Monet benchmark baseline under documented settings.
- [ ] Freeze the reproduced baseline with a Git tag.
- [ ] Locate the exact code path that creates/updates continuous latent visual states.
- [ ] Verify latent-state shape, positions, count, and generation behavior.
- [ ] Start V0 only after the above checks pass.

## Active Method Version
None. V0 has not started.

## Current Experiment
None.

## Known Issues
None recorded yet.

## Next Action
Inspect and establish the official Monet baseline implementation. Do not implement positive/negative views or latent losses before baseline reproduction.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
