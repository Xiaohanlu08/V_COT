# Current State

## Project Stage
Baseline environment bootstrap / pre-inference.

## Current Objective
Establish a reproducible Monet baseline before implementing any new latent-supervision method.

## Current Branch
`main`

## Selected Upstream Baseline
- Monet repository: `https://github.com/NOVAglow646/Monet.git`
- Pinned commit: `08939998d3d643a73a316e349faa34f420429153`
- Primary checkpoint for first reproduction: `NOVAglow646/Monet-7B`
- Baseline specification: `BASELINE.md`

## Last Verified State
The V_COT repository has been initialized and the official Monet upstream source/commit has been pinned in project documentation. A mirrored 4 x RTX 3090 bootstrap path has been added.

No server-side Monet environment, model download, inference result, benchmark result, or modified method has yet been verified.

## Current Task
Run `scripts/bootstrap_3090.sh` on the server and verify:
- the `vcot` environment is created successfully;
- Monet source resolves to the pinned SHA;
- Monet-7B is downloaded successfully through the configured Hugging Face mirror;
- PyTorch sees all available GPUs correctly.

## Next Milestones
- [x] Select the exact Monet implementation to use as the baseline.
- [x] Record the upstream repository URL and upstream commit SHA.
- [ ] Bootstrap the 3090 environment successfully.
- [ ] Download the official Monet-7B checkpoint.
- [ ] Reproduce official inference on at least one provided example.
- [ ] Observe/verify latent-mode generation behavior.
- [ ] Reproduce the selected Monet benchmark baseline under documented settings.
- [ ] Freeze the reproduced baseline with a Git tag.
- [ ] Locate the exact code path that creates/updates continuous latent visual states.
- [ ] Verify latent-state shape, positions, count, and generation behavior.
- [ ] Start V0 only after the above checks pass.

## Hardware Plan
- Development/debug/V0 pilot: 4 x RTX 3090.
- Full-scale or RL/VLPO experiments: H200 when justified by memory/runtime.

This is a resource-allocation decision, not a change to the scientific goal.

## Active Method Version
None. V0 has not started.

## Current Experiment
Environment bootstrap only; not yet an experiment.

## Known Issues
- Official Monet SFT scripts are written for 8 GPUs with DeepSpeed ZeRO-2; they will not be treated as a drop-in 4 x RTX 3090 recipe.
- The project will not change package versions casually after bootstrap failures because Monet uses customized Transformers/vLLM code.

## Next Action
On the server, pull this repository and run the bootstrap procedure in `SETUP_3090.md`. Return the requested GPU, SHA, Python, PyTorch, and model-directory output before attempting inference or training.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
