import os
import re
from pathlib import Path


def clean_latent_text(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)


def main():
    import inference.apply_vllm_monet  # patch before importing vLLM
    import PIL.Image
    from transformers import AutoProcessor
    from inference.load_and_gen_vllm import (
        vllm_generate,
        vllm_mllm_init,
        vllm_mllm_process_batch_from_messages,
    )

    model_path = os.environ['MONET_MODEL_DIR']
    image_path = Path('images/example_question.png')

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    start_id = processor.tokenizer.convert_tokens_to_ids('<abs_vis_token>')
    end_id = processor.tokenizer.convert_tokens_to_ids('</abs_vis_token>')

    print('=============== TOKEN CHECK ===============')
    print('start token id:', start_id)
    print('end token id:', end_id)
    print('expected start id:', os.environ.get('LATENT_START_ID', '151666'))
    print('expected end id:', os.environ.get('LATENT_END_ID', '151667'))
    assert start_id == 151666, f'unexpected start token id: {start_id}'
    assert end_id == 151667, f'unexpected end token id: {end_id}'

    print('\n[VERIFY] initializing Monet/vLLM...')
    mllm, sampling_params = vllm_mllm_init(
        model_path,
        tp=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
    )

    question = (
        'Question: Which car has the longest rental period? The choices are listed below:\n'
        '(A) DB11 COUPE.\n'
        '(B) V12 VANTAGES COUPES.\n'
        '(C) VANQUISH VOLANTE.\n'
        '(D) V12 VOLANTE.\n'
        '(E) The image does not feature the time.\n\n'
        'Diagnostic instruction: before answering, use the model latent visual reasoning mode. '
        'Begin your response exactly with the special token <abs_vis_token>. '
        'Do not output any ordinary word before that token. After the latent reasoning segment, '
        'give the final answer in \\boxed{}.'
    )

    conversations = [[{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': question},
            {'type': 'image', 'image': PIL.Image.open(image_path).convert('RGB')},
        ],
    }]]

    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    raw = output[0].outputs[0].text
    cleaned = clean_latent_text(raw)

    print('\n================ RAW OUTPUT ================')
    print(raw)
    print('\n============== CLEANED OUTPUT ==============')
    print(cleaned)
    print('\n=============== LATENT CHECK ===============')
    print('contains <abs_vis_token>:', '<abs_vis_token>' in raw)
    print('contains </abs_vis_token>:', '</abs_vis_token>' in raw)
    print('LATENT_SIZE:', os.environ.get('LATENT_SIZE'))

    if '<abs_vis_token>' in raw:
        print('\n[LATENT TRIGGER PASS] The model emitted the latent-start token and Monet runner entered the latent path.')
    else:
        print('\n[LATENT TRIGGER NOT OBSERVED] Token wiring is correct, but the model did not emit the trigger under this diagnostic prompt.')


if __name__ == '__main__':
    main()
