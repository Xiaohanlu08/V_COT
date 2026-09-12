import os
from pathlib import Path


def main() -> None:
    import PIL.Image
    from transformers import AutoProcessor
    from inference.load_and_gen_vllm import (
        vllm_generate,
        vllm_mllm_init,
        vllm_mllm_process_batch_from_messages,
    )

    model_path = os.environ["MONET_MODEL_DIR"]
    image_path = Path("images/example_question.png")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    start_id = processor.tokenizer.convert_tokens_to_ids("<abs_vis_token>")
    end_id = processor.tokenizer.convert_tokens_to_ids("</abs_vis_token>")

    print("=============== TOKEN CHECK ===============")
    print("start token id:", start_id)
    print("end token id:", end_id)
    assert start_id == int(os.environ["LATENT_START_ID"])
    assert end_id == int(os.environ["LATENT_END_ID"])

    print("\n[FORCE] initializing spawn-safe Monet/vLLM...")
    mllm, sampling_params = vllm_mllm_init(
        model_path,
        tp=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
    )

    # Engineering-only path test: force every sampled token in this short
    # request to the latent-start token using vLLM V1's built-in
    # allowed_token_ids mask. This is not a benchmark setting.
    sampling_params.allowed_token_ids = [start_id]
    sampling_params.max_tokens = 4
    sampling_params.temperature = 0.0

    conversations = [[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Question: Which car has the longest rental period? "
                    "This is an engineering path test; no natural-trigger "
                    "claim is being made."
                ),
            },
            {
                "type": "image",
                "image": PIL.Image.open(image_path).convert("RGB"),
            },
        ],
    }]]

    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    candidate = output[0].outputs[0]
    raw = candidate.text
    token_ids = list(candidate.token_ids)

    print("\n================ RAW OUTPUT ================")
    print(raw)
    print("\n=============== TOKEN IDS ==================")
    print(token_ids)
    print("\n============= FORCED PATH CHECK ============")
    print("first token is latent start:", bool(token_ids and token_ids[0] == start_id))
    print("all generated ids are latent start:", bool(token_ids and all(t == start_id for t in token_ids)))
    print("generated token count:", len(token_ids))
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))

    assert token_ids, "No tokens were generated."
    assert token_ids[0] == start_id, token_ids
    assert len(token_ids) >= 2, token_ids

    print(
        "\n[FORCED LATENT PATH PASS] The request sampled LATENT_START_ID. "
        "If the log also shows the spawned worker was patched by "
        "[VCOT_MONET_SITE] and the Monet runner reports start/end/size, "
        "then subsequent decode steps execute the runner's active latent path."
    )


if __name__ == "__main__":
    main()
