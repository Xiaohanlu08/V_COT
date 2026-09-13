#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: 14_make_fixed_trigger_runner.py <runner.py>")

    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")

    anchor = '''                st = self.latent_state.setdefault(req_id, {"active": False, "pending": None, "current_len": 0})
                # Detect boundaries from sampled ids of this step
                gen_ids = valid_sampled_token_ids[i]'''

    replacement = '''                st = self.latent_state.setdefault(req_id, {"active": False, "pending": None, "current_len": 0})
                # V_COT counterfactual replay control: when enabled, override
                # only the first sampled token of each request with the Monet
                # latent-start token. Subsequent sampling is untouched.
                if (os.environ.get("VCOT_FORCE_LATENT_START_ONCE") == "1"
                        and not st.get("_vcot_forced_start_once", False)):
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

    print("FIXED_TRIGGER_RUNNER_PATCH_PASS=True")
    print("patched_runner:", path)


if __name__ == "__main__":
    main()
