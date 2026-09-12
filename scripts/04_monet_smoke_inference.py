import os
import re
from pathlib import Path


def clean_latent_text(s: str) -> str:
    pattern = re.compile(r"(<abs_vis_token>)(.*?)(</abs_vis_token>)", flags=re.DOTALL)
    return pattern.sub(r"\1<latent>\3", s)


def main() -> None:
    # Monet must patch vLLM before vLLM itself is imported.
    import inference.apply_vllm_monet  # noqa: F401
    import PIL.Image
    from transformers import AutoProcessor
    from inference.load_and_gen_vllm import (
        vllm_generate,
        vllm_mllm_init,
        vllm_mllm_process_batch_from_messages,
    )

    model_path = os.environ["MONET_MODEL_DIR"]
    image_path = Path("images/example_question.png")

    print("[SMOKE] initializing Monet/vLLM...")
    mllm, sampling_params = vllm_mllm_init(
        model_path,
        tp=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
    )

    print("[SMOKE] loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    conversations = [[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Question: Which car has the longest rental period? The choices are listed below:\n"
                    "(A) DB11 COUPE.\n"
                    "(B) V12 VANTAGES COUPES.\n"
                    "(C) VANQUISH VOLANTE.\n"
                    "(D) V12 VOLANTE.\n"
                    "(E) The image does not feature the time. "
                    "Put your final answer in \\boxed{}."
                ),
            },
            {
                "type": "image",
                "image": PIL.Image.open(image_path).convert("RGB"),
            },
        ],
    }]]

    print("[SMOKE] preparing multimodal input...")
    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)

    print("[SMOKE] generating...")
    output = vllm_generate(inputs, sampling_params, mllm)
    raw = output[0].outputs[0].text
    cleaned = clean_latent_text(raw)

    print("\n================ RAW OUTPUT ================")
    print(raw)
    print("\n============== CLEANED OUTPUT ==============")
    print(cleaned)
    print("\n=============== LATENT CHECK ===============")
    print("contains <abs_vis_token>:", "<abs_vis_token>" in raw)
    print("contains </abs_vis_token>:", "</abs_vis_token>" in raw)
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))

    assert raw.strip(), "Model returned an empty output."
    print("\n[SMOKE PASS] Monet produced a non-empty output.")


if __name__ == "__main__":
    # Required for vLLM's spawn-based worker startup.
    main()
