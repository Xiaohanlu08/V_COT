#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: 12_make_tensor_capture_runner.py <runner.py>")

    path = Path(sys.argv[1]).resolve()
    text = path.read_text(encoding="utf-8")

    anchor = '''                    st["pending"] = last_token_h[i].detach()\n                    st["current_len"] +=1'''

    replacement = '''                    st["pending"] = last_token_h[i].detach()\n                    # V_COT observation-only instrumentation: dump the exact\n                    # pending hidden-state vector that Monet will feed into the\n                    # next latent decode step. Restrict writes to global rank 0.\n                    if os.environ.get("VCOT_LATENT_TENSOR_DUMP") == "1":\n                        rank0 = (\n                            (not torch.distributed.is_initialized())\n                            or torch.distributed.get_rank() == 0\n                        )\n                        dump_dir = os.environ.get("VCOT_LATENT_DUMP_DIR")\n                        if rank0 and dump_dir:\n                            os.makedirs(dump_dir, exist_ok=True)\n                            dump_step = int(st.get("current_len", 0)) + 1\n                            dump_tensor = (\n                                st["pending"].detach()\n                                .to(device="cpu", dtype=torch.float32)\n                                .contiguous()\n                            )\n                            torch.save(\n                                dump_tensor,\n                                os.path.join(\n                                    dump_dir,\n                                    f"latent_step_{dump_step:02d}.pt",\n                                ),\n                            )\n                    st["current_len"] +=1'''

    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one Monet pending-state anchor, found {count}. "
            "Refusing to patch an unexpected runner."
        )

    text = text.replace(anchor, replacement, 1)
    path.write_text(text, encoding="utf-8")

    print("TENSOR_CAPTURE_RUNNER_PATCH_PASS=True")
    print("patched_runner:", path)


if __name__ == "__main__":
    main()
