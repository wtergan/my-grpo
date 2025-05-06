"""
Light-weight inference/evaluation.
"""

from __future__ import annotations
import argparse, json, sys, time, pathlib, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import data_utils as du
import yaml


# ===============================================================================
# LOADING OF MODEL AND TOKENIZER
# ===============================================================================
def load_model(model_name, ckpt, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from{ckpt}")
    model.to(device).eval()
    return model, tokenizer

# ===============================================================================
# GENERATION OF COMPLETIONS
# ===============================================================================
def generate(model, tokenizer, prompt, max_new_tokens, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

# ===============================================================================
# MAIN TESTING ROUTINE
# ===============================================================================
def main(test_cfg, model_cfg, data_cfg):
    model = AutoModelForCausalLM.from_pretrained(
        test_cfg.get('model_name', model_cfg['model_path']),
        torch_dtype=torch.bfloat16 if model_cfg['dtype'] == 'bfloat16' else torch.float16 if model_cfg['dtype'] == 'float16' else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(test_cfg.get('model_name', model_cfg['model_path']))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
    model.to(model_cfg['device']).eval()

    prompts_file = test_cfg.get('prompts')
    #task = test_cfg.get('task', data_cfg['data_path'])
    out_file = test_cfg.get('out')
    max_new_tokens = test_cfg.get('max_new_tokens', 128)

    # BATCH MODE:
    if prompts_file:
        if not os.path.exists(prompts_file):
            print(f"Error: Prompts file not found: {prompts_file}")
            return
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            print(f"Warning: Prompts file {prompts_file} is empty.")
            if out_file:
                with open(out_file, 'w') as f:
                    json.dump([], f) 
            return
    else: 
        # REPL MODE:
        prompts = []
        try:
            prompt_text = input('Prompt: ')
            if prompt_text.strip(): 
                prompts.append(prompt_text.strip())
            else: 
                sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
        except EOFError:
            sys.exit(0)

    if not prompts: 
        if out_file:
            if prompts_file: 
                 with open(out_file, 'w') as f:
                    json.dump([], f)
        return

    output_records = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model_cfg['device'])
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        output_records.append({'prompt': prompt, 'completion': completion})

    if out_file is not None:
        with open(out_file, 'w') as f:
            json.dump(output_records, f, indent=2)
    else:
        print(json.dumps(output_records, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    main(test_cfg, model_cfg, data_cfg)