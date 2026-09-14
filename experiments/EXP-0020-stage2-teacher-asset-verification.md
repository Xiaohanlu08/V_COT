# EXP-0020 — Public Stage-2 Teacher Asset Acquisition and Structural Verification

**Status:** COMPLETED

**Purpose:** Acquire only the minimum additional published checkpoint justified after the Stage-3 runtime-contract probe: the official Stage-2 teacher used by Monet Stage-3 precompute. Verify the downloaded safetensors structurally before any real-teacher metric probe.

**Asset:**
```text
NOVAglow646/Monet-SFT-7B/stage2
local: models/Monet-SFT-7B-stage2
```
Stage-1 and Stage-3 were not downloaded.

**Transport:** Four safetensors shards downloaded through exact resumable HTTP byte ranges. Every weight chunk required a valid `Content-Range` response.

**Step-27 verifier issue:** The initial final check compared the sum of complete `.safetensors` file sizes against `model.safetensors.index.json:metadata.total_size` and reported:
```text
local file bytes: 16578766160
index total_size: 16578684928
raw difference: 81232 bytes
```
This equality is not valid because `metadata.total_size` represents tensor payload bytes, while each safetensors file also contains the 8-byte header-length prefix and JSON header/padding.

**Corrected Step-27b verification:** Parsed all local safetensors headers directly and required:
1. sum of tensor `data_offsets` payload bytes equals index `metadata.total_size`;
2. each shard container size equals `8 + header_len + max(data_offset_end)`;
3. no overlapping tensor byte ranges;
4. tensor names in shard headers match the index `weight_map` per shard and globally.

**Verified result:**
```text
index_total_weight_bytes:       16578684928
safetensors_payload_bytes:      16578684928
local_file_bytes:               16578766160
container_header_overhead:             81232
payload_matches_index_total:    true
all_shard_container_sizes_exact:true
all_shards_no_overlapping_ranges:true
per_shard_index_names_match:    true
global_index_names_match:       true
```

**Per-shard structure:**
```text
model-00001-of-00004.safetensors
  file=4965419112 payload=4965368832 overhead=50280 tensors=459
model-00002-of-00004.safetensors
  file=4991495816 payload=4991480832 overhead=14984 tensors=131
model-00003-of-00004.safetensors
  file=4932751040 payload=4932737024 overhead=14016 tensors=122
model-00004-of-00004.safetensors
  file=1689100192 payload=1689098240 overhead=1952 tensors=17
```

**SHA-256 manifest:**
```text
model-00001-of-00004.safetensors  daa156afaf34fed7be870dbccdd12db0e3e92187c9d755f31653c4ccb6ce2954
model-00002-of-00004.safetensors  5140dfeab39fe95c784bc8bfd4e3279b1ff2059e376ea58aebedd3bb290e5799
model-00003-of-00004.safetensors  bd55afc3a00da7cd44099e9e0fc21a1d535a62ee039e79dacc050afc64117aab
model-00004-of-00004.safetensors  debf05227df9795774a51a1fc49e1b980e731999861565f1cb986408c4514d73
```

**Saved verification:**
`results/v0_prep/stage2_asset_structural_verification.json`

**Conclusion:** KEEP. The Stage-2 checkpoint is structurally verified and must not be re-downloaded. The next experiment may use it to generate real official all-layer teacher targets on a small frozen Visual_CoT cohort.
