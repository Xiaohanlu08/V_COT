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

The target 4 x RTX 3090 server has **restricted overseas access**, not a full Internet outage.

Verified on 2026-09-12:
- TUNA PyPI mirror is reachable from the server (`HTTP/2 200`).
- `https://hf-mirror.net` is reachable from the server (`HTTP/2 200`).
- direct GitHub access fails with GnuTLS handshake errors.
- `gh-proxy.com` and `ghfast.top` also fail with GnuTLS handshake errors from this server.
- `gitclone.com` returned HTTP 502 during the test.

Therefore GitHub source transport must be relayed through an Internet-connected machine, while Python packages and model checkpoints should be downloaded directly on the GPU server through the verified mirrors.

No server-side Monet environment, model download, inference result, benchmark result, or modified method has yet been verified.

## Current Task
1. Transfer the current `V_COT` repository to `~/work/V_COT` through an Internet-connected machine.
2. Transfer Monet pinned at commit `08939998d3d643a73a316e349faa34f420429153` to `~/work/V_COT/third_party/Monet`.
3. Run `scripts/01_prepare_restricted_env.sh` on the server.
4. Inspect GPU driver/CUDA compatibility before installing Monet's heavy requirements.

## Next Milestones
- [x] Select the exact Monet implementation to use as the baseline.
- [x] Record the upstream repository URL and upstream commit SHA.
- [x] Characterize the server network as restricted-overseas rather than fully offline.
- [x] Verify TUNA PyPI availability from the server.
- [x] Verify HF mirror availability from the server.
- [ ] Transfer V_COT source to the server.
- [ ] Transfer pinned Monet source to `third_party/Monet`.
- [ ] Create the `vcot` Python 3.10 environment through domestic mirrors.
- [ ] Verify GPU driver/CUDA compatibility for the Monet/vLLM dependency set.
- [ ] Install the pinned Monet requirements.
- [ ] Download the official Monet-7B checkpoint through `hf-mirror.net`.
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
- Direct GitHub access from the target server is unavailable; GitHub must not be part of the server-side bootstrap path.
- Official Monet SFT scripts are written for 8 GPUs with DeepSpeed ZeRO-2; they will not be treated as a drop-in 4 x RTX 3090 recipe.
- Monet pins `vllm==0.10.0` and uses customized Transformers/vLLM code. Heavy package installation will be done only after checking the server GPU driver and CUDA compatibility.
- The project will not change package versions casually after bootstrap failures.

## Next Action
Relay V_COT and the pinned Monet source from an Internet-connected machine to the server. Then run `scripts/01_prepare_restricted_env.sh` and return the complete terminal output before installing Monet requirements or downloading the checkpoint.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
