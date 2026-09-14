# EXP-0022 — Public Stage-1 Student Initialization Asset Verification

**Status:** COMPLETED; VERIFIED

**Date:** 2026-09-14

**Purpose:** Acquire and verify the public Stage-1 checkpoint used as the student initialization for official Stage-3 training, so the real-teacher metric probe can be rerun without the final-Monet checkpoint mismatch present in EXP-0021.

## Asset
```text
NOVAglow646/Monet-SFT-7B/stage1
local: models/Monet-SFT-7B-stage1
```
Pinned public upload commit:
```text
fb8f3da99888c271a1732fb1ad6594f93f82a652
```
Stage-3 was not downloaded. Existing Stage-2 was left untouched.

## Verification
All four safetensors shards completed and passed both structural and upstream hash verification.

```text
index_total_weight_bytes: 16578684928
safetensors_payload_bytes: 16578684928
local_container_file_bytes: 16578766160
container_header_overhead_bytes: 81232
payload_matches_index_total: true
all_shard_container_sizes_exact: true
all_shards_nonoverlapping: true
per_shard_index_names_match: true
global_index_names_match: true
all_upstream_sha256_match: true
```

Per-shard structure:
```text
model-00001-of-00004.safetensors  file=4965419112  payload=4965368832  overhead=50280  tensors=459
model-00002-of-00004.safetensors  file=4991495816  payload=4991480832  overhead=14984  tensors=131
model-00003-of-00004.safetensors  file=4932751040  payload=4932737024  overhead=14016  tensors=122
model-00004-of-00004.safetensors  file=1689100192  payload=1689098240  overhead=1952   tensors=17
```

## Conclusion
**KEEP.** Stage-1 is now locally available and verified as the clean checkpoint discriminator required by the EXP-0021 protocol.

## Next action
Rerun the exact frozen EXP-0021 8-row real-Stage-2-teacher probe with Stage-1 as the student checkpoint. Change no cohort rows, evidence operator, metric, layer reduction, or viability threshold.

Decision rule remains unchanged:
- PASS if explicit-`dim=-1` positive gaps are at least `6/8` and median gap is positive;
- FAIL otherwise, in which case reject teacher-anchored explicit-`dim=-1` natural ranking as the first V0 evidence metric rather than post-hoc tuning the probe.
