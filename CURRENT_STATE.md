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
The V_COT repository has been initialized and the official Monet upstream source/commit has been pinned in project documentation.

The target 4 x RTX 3090 server is **not fully offline**. It can access ordinary mainland/domestic Internet resources and can create/install Python environments, but direct overseas access such as GitHub is unavailable or unreliable. Therefore the active setup guide is now `RESTRICTED_NETWORK_SETUP.md`; `OFFLINE_SETUP.md` is retained only as a fallback for a truly air-gapped machine.

No server-side Monet environment, model download, inference result, benchmark result, or modified method has yet been verified.

## Current Task
Use a split transport strategy:
- transfer small GitHub source trees (`V_COT` and pinned Monet) to the server through an Internet-connected machine or a future trusted domestic Git mirror;
- create the Python environment directly on the server via domestic Conda/PyPI mirrors;
- download `NOVAglow646/Monet-7B` directly on the server via `HF_ENDPOINT=https://hf-mirror.net` if reachable.

## Next Milestones
- [x] Select the exact Monet implementation to use as the baseline.
- [x] Record the upstream repository URL and upstream commit SHA.
- [x] Characterize the server network as restricted-overseas rather than fully offline.
- [ ] Transfer V_COT source to the server.
- [ ] Transfer pinned Monet source to `third_party/Monet`.
- [ ] Bootstrap the `vcot` Python environment successfully through domestic mirrors.
- [ ] Download the official Monet-7B checkpoint through the configured Hugging Face mirror, or transfer it if the mirror is inaccessible.
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
- Direct GitHub access from the target server fails with GnuTLS handshake errors; GitHub should not be part of the server-side bootstrap path.
- Official Monet SFT scripts are written for 8 GPUs with DeepSpeed ZeRO-2; they will not be treated as a drop-in 4 x RTX 3090 recipe.
- The project will not change package versions casually after bootstrap failures because Monet uses customized Transformers/vLLM code.

## Next Action
Transfer the small V_COT and Monet source trees to the server, then follow `RESTRICTED_NETWORK_SETUP.md` to create the environment and download model weights through domestic mirrors. Return the requested GPU, Python, PyTorch, Monet SHA, and model-directory output before attempting inference or training.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
