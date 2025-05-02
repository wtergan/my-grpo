"""
Light-weight inference/evaluation.
"""

from __future__ import annotations
import argparse, json, sys, time, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import data_utils as du


# ===============================================================================
# ARGUMENT PARSER
# ===============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO INFERENCE")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--ckpt",       default=None, help="Path to the .pt checkpoint")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompts",    default=None, 
                   help="Text/JSONL file with one prompt per line. If omitted: REPL mode.")
    p.add_argument("--task",       default="gsm8k", 
                   help="Task name to compute reward (if targets available)")
    p.add_argument("--out",        default=None, 
                   help="Where to dump JSONL of {prompt,completion,reward}")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p

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
def main():
    args = build_parser().parse_args()
    model, tokenizer = load_model(args.model_name, args.ckpt, args.device)

    # REPL MODE: Meaning no prompt file(s) were provided, allow user to input prompts:
    if args.prompts is None:
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
            completion = generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
            delta_t = time.time() - t0
            print(f"\n{completion}\n took {delta_t:.2f}seconds\n")
    
    # BATCH MODE: Meaning prompt file(s) were provided, lets process them:
    else:
        prompts = [line.rstrip("\n") for line in open(args.prompts)]
        targets = None
        if args.targets:
            targets = [line.rstrip("\n") for line in open(args.targets)]
            assert len(prompts) == len(targets), "prompts and targets must have the same length."
        
        output_records = []
        reward_fn = lambda preds, tgts: du.compute_binary_reward(preds, tgts, args.task)
        for i, prompt in enumerate(prompts):
            completion = generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
            rec = {"prompt": prompt, "completion": completion}
            if targets:
                rec["target"] = targets[i]
                rec["reward"] = float(reward_fn([completion], [targets[i]])[0])
            output_records.append(rec)
            print(f"[{i:>3}] {rec['reward'] if 'reward' in rec else ''}\t{rec['completion'][:80]}")

        if args.out:
            with open(args.out, "w") as f:
                for rec in output_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Wrote {len(output_records)} lines to {args.out}")

if __name__ == "__main__":
    main()