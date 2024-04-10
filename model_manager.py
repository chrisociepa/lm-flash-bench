import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelManager:

    def __init__(self, config):
        self.model = None
        self.tokenizer = None
        self.use_chat_template = True
        self.model_config = {
            'compile': True,
            'dtype': 'bfloat16',
            'device': 'cuda:0',
            'use_chat_template': True,
            'gen_args': {
                'max_new_tokens': 200,
                'top_k': 50,
                'do_sample': False,
                'top_p': 1.0,
                'num_beams': 1,
                'bos_token_id': 1,
                'eos_token_id': 2,
                'pad_token_id': 0,
                'repetition_penalty': 1.1
            }
        }
        
        if config['model_config_path']:
            with open(config['model_config_path'], "r", encoding="utf-8") as f:
                self.model_config = json.load(f)
                
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.model_config.get("dtype", "bfloat16")]
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.model = AutoModelForCausalLM.from_pretrained(config['model_path'], torch_dtype=dtype, device_map=self.model_config.get("device", "auto"))
        if self.model_config.get("compile", True):
            self.model = torch.compile(self.model)
        
    def generate(self, prompt):
        gen_args = self.model_config['gen_args']
        if self.model_config['use_chat_template']:
            chat_history = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            prompt = self.tokenizer.apply_chat_template(chat_history, tokenize=False)
        else:
            assert isinstance(prompt, str)
        input_ids = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(self.model.device)
        input_tokens_count = len(input_ids[0])
        input_chars_count = len(prompt)
        start_time = time.time()
        output = self.model.generate(inputs=input_ids, **gen_args)
        output = output[0][len(input_ids[0]):]
        elapsed_time = time.time() - start_time
        decoded_output = self.tokenizer.decode(output, skip_special_tokens=True)
        output_tokens_count = len(output)
        output_chars_count = len(decoded_output)
        return {
            "input_tokens": input_tokens_count,
            "input_chars": input_chars_count,
            "output_tokens": output_tokens_count,
            "output_chars": output_chars_count,
            "generation_time": elapsed_time,
            "model_response": decoded_output
        }
