"""CPU-only check of the pinned backend's actual text collator; no model weights loaded."""
import json
import sys

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model import load_tokenizer

module = load_tokenizer(ModelArguments(model_name_or_path=sys.argv[1], trust_remote_code=False))
tokenizer, processor = module['tokenizer'], module['processor']
assert processor is not None, 'Qwen processor missing'
template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template='qwen3_5_nothink'))
messages = [{'role': 'user', 'content': 'Write a one-bit wire assignment.'},
            {'role': 'assistant', 'content': 'assign y = a;'}]
messages = template.mm_plugin.process_messages(messages, [], [], [], processor)
prompt, answer = template.encode_oneturn(tokenizer, messages)
tokens = prompt + answer
collator = MultiModalDataCollatorForSeq2Seq(tokenizer=tokenizer, processor=processor, template=template)
batch = collator([{'input_ids': tokens, 'attention_mask': [1] * len(tokens),
                   'labels': [-100] * len(prompt) + answer}])
print(json.dumps({'processor': type(processor).__name__,
                  'batch_shapes': {k: list(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}}))
