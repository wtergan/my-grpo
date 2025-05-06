"""
Light-weight inference/evaluation.
"""

from __future__ import annotations
import argparse, json, sys, time, pathlib
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
def main(config):
    test_cfg = config['testing']
    model, tokenizer = load_model(test_cfg['model_name'], test_cfg['ckpt'], test_cfg['device'])

    # REPL MODE: Meaning no prompt file(s) were provided, allow user to input prompts:
    if test_cfg['prompts'] is None:
        print("Enter prompt(s) (empty line to quit):")
        while True:
            try:
                prompt = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("No prompt entered. Exiting.")
                break
            if not prompt:
                print("No prompt entered. Exiting.")
                break
            t0 = time.time()
            completion = generate(model, tokenizer, prompt, test_cfg['max_new_tokens'], test_cfg['device'])
            delta_t = time.time() - t0
            print(f"\n{completion}\n took {delta_t:.2f}seconds\n")
    
    # BATCH MODE: Meaning prompt file(s) were provided, lets process them:
    else:
        prompts = [line.rstrip("\n") for line in open(test_cfg['prompts'])]
        targets = None
        if 'targets' in test_cfg:
            targets = [line.rstrip("\n") for line in open(test_cfg['targets'])]
            assert len(prompts) == len(targets), "prompts and targets must have the same length."
        
        output_records = []
        reward_fn = lambda preds, tgts: du.compute_binary_reward(preds, tgts, test_cfg['task'])
        for i, prompt in enumerate(prompts):
            completion = generate(model, tokenizer, prompt, test_cfg['max_new_tokens'], test_cfg['device'])
            rec = {"prompt": prompt, "completion": completion}
            if targets:
                rec["target"] = targets[i]
                rec["reward"] = float(reward_fn([completion], [targets[i]])[0])
            output_records.append(rec)
            print(f"[{i:>3}] {rec['reward'] if 'reward' in rec else ''}\t{rec['completion'][:80]}")

        if 'out' in test_cfg:
            with open(test_cfg['out'], "w") as f:
                for rec in output_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Wrote {len(output_records)} lines to {test_cfg['out']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)