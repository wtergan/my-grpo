"""
Relatively simple GRPO training procedure.
"""

from __future__ import annotations
from pathlib import Path
import argparse, random, os, math, json, time
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from tqdm.auto import tqdm
import data_utils as du
import grpo_core as gc

# ===============================================================================
# ARGUMENT PARSER
# ===============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO TRAINER")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--task",       default="gsm8k")
    p.add_argument("--batch_n",    type=int, default=8,   help="N (prompts)")
    p.add_argument("--gens_m",     type=int, default=4,   help="M (answers per prompt)")
    p.add_argument("--steps",      type=int, default=1_000)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--warmup",     type=int,   default=100)
    p.add_argument("--eval_every", type=int,   default=200)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--fp16",       action="store_true")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inference",  action="store_true", help="Use device_map='auto' for inference mode")
    return p

# ===============================================================================
# SEED SETUP
# ===============================================================================
def seed_setup(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===============================================================================
# EVALUATION FUNCTION
# ===============================================================================
@torch.no_grad()
def evaluation(model, tokenizer, val_prompts, val_targets, reward_fn,
               num_generations, max_new_tokens, device):
    """Greedy single completion per prompt."""
    model.eval()
    outs = []
    for i in range(0, len(val_prompts), 32):
        slice_prompts = val_prompts[i:i+32]
        enc = tokenizer(slice_prompts, return_tensors="pt",
                        padding=True, truncation=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id)
        cmp = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        outs.extend(cmp)
    rewards = reward_fn(outs, val_targets)
    return rewards.mean().item()

# ===============================================================================
# MAIN TRAINING LOOP
# ===============================================================================
def main():
    args = build_argparser().parse_args()
    seed_setup(args.seed)
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    # Model and token initialization:
    model_kwargs = {
        "torch_dtype": torch.float16 if args.fp16 else torch.float32,
    }
    if args.inference and args.device.startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
    model.to(args.device).train()

    # Dataset loading:
    train_ds, val_ds = du.load_task_dataset(args.task)
    train_prompts = du.build_prompts(train_ds, args.task)
    val_prompts = du.build_prompts(val_ds, args.task)
    train_targets = du.target_extraction(train_ds, args.task)
    val_targets = du.target_extraction(val_ds, args.task)
    reward_fn = lambda preds, tgts: du.compute_binary_reward(preds, tgts, args.task)

    # Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup, num_training_steps=args.steps)
    
    # Logging setup:
    t_board = SummaryWriter(log_dir=f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    step_bar = tqdm(range(1, args.steps + 1), desc="train-step")

    # Training Loop:
    for step in step_bar:
        # Sampling N prompts:
        idx = random.sample(range(len(train_prompts)), args.batch_n)
        prompts = [train_prompts[i] for i in idx]
        targets = [train_targets[i] for i in idx]

        # Forward-propagation:
        loss, diag = gc.grpo_step(
            model, tokenizer, prompts, targets, reward_fn,
            num_generations=args.gens_m,
            max_new_tokens=64,
            device=args.device,
        )

        # Back-propagation:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging process:
        step_bar.set_postfix(loss=float(loss), reward=diag["raw_reward_mean"])
        t_board.add_scalar("train/loss", float(loss), step)
        t_board.add_scalar("train/r_mean", diag["raw_reward_mean"], step)

        # Evaluation and checkpointing:
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluation(model, tokenizer, val_prompts, val_targets, reward_fn, 
                          args.gens_m, 64, args.device)
            t_board.add_scalar("val/accuracy", acc, step)
            torch.save(model.state_dict(), f"{args.save_dir}/model_step_{step}.pt")
            model.train()
        
    # Close logging:
    t_board.close()
    
# ===============================================================================
# SCRIPT ENTRY POINT
# ===============================================================================
if __name__ == "__main__":
    main()
