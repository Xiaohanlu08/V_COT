#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: 18_make_confirmatory_runner.py <runner.py>")

    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")

    anchor = '''                st = self.latent_state.setdefault(req_id, {"active": False, "pending": None, "current_len": 0})
                # Detect boundaries from sampled ids of this step
                gen_ids = valid_sampled_token_ids[i]'''

    replacement = '''                st = self.latent_state.setdefault(req_id, {"active": False, "pending": None, "current_len": 0})
                # V_COT confirmatory replay control. Before overriding the first
                # sampled token, record what greedy decoding would naturally
                # have selected at this fixed-prefix boundary. This lets the
                # original image verify that the replay lands exactly at its
                # natural latent-entry point. Then override only this first
                # token with the Monet latent-start token; later sampling is
                # untouched.
                if (os.environ.get("VCOT_FORCE_LATENT_START_ONCE") == "1"
                        and not st.get("_vcot_forced_start_once", False)):
                    pre_force_first = (
                        valid_sampled_token_ids[i][0]
                        if valid_sampled_token_ids[i] else -1
                    )
                    rank0 = (
                        (not torch.distributed.is_initialized())
                        or torch.distributed.get_rank() == 0
                    )
                    preforce_dir = os.environ.get("VCOT_PREFORCE_DUMP_DIR")
                    if rank0 and preforce_dir:
                        os.makedirs(preforce_dir, exist_ok=True)
                        with open(
                            os.path.join(preforce_dir, "preforce_token.txt"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(str(int(pre_force_first)))
                    valid_sampled_token_ids[i] = [self.latent_start_id]
                    st["_vcot_forced_start_once"] = True
                # Detect boundaries from sampled ids of this step
                gen_ids = valid_sampled_token_ids[i]'''

    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one Monet latent-state anchor, found {count}. "
            "Refusing to patch an unexpected runner."
        )

    text = text.replace(anchor, replacement, 1)
    path.write_text(text, encoding="utf-8")

    print("CONFIRMATORY_RUNNER_PATCH_PASS=True")
    print("patched_runner:", path)


if __name__ == "__main__":
    main()
