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
import contextlib
import yaml

# =============================================================================
# MIXED PRECISION SETUP
# =============================================================================
def mixed_precision_env(device_name, dtype="bfloat16"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    context = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return {
        'device': device_name,
        'context': context,
        'device_type': device_type,
    }

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
def main(config):
    train_cfg = config['training']
    seed_setup(train_cfg['random_seed'])
    Path(train_cfg['ckpt_dir']).mkdir(exist_ok=True, parents=True)

    # Model and token initialization:
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config['model']['dtype'] == 'bfloat16' else torch.float16 \
            if config['model']['dtype'] == 'float16' else torch.float32,
    }
    if config['model'].get('inference', False) and config['model']['device'].startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_path'],
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
    model.to(config['model']['device']).train()

    # Set up mixed precision context:
    env = mixed_precision_env(config['model']['device'], dtype=config['model']['dtype'])
    context = env['context']

    # GradScaler for mixed-precision (only CUDA):
    scaler = torch.amp.GradScaler(device=config['model']['device']) if config['model']['device'].startswith('cuda') else None

    # Load reference model if KL is enabled:
    ref_model = None
    if train_cfg.get('kl_beta', 0) > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            train_cfg.get('ref_model_name', config['model']['model_path']),
            torch_dtype=torch.bfloat16 if config['model']['dtype'] == 'bfloat16' else torch.float16 if config['model']['dtype'] == 'float16' else torch.float32,
            device_map="auto" if config['model']['device'].startswith("cuda") else None
        )
        ref_model.to(config['model']['device']).eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # Dataset loading:
    train_ds, val_ds = du.load_task_dataset(config['data']['data_path'])
    train_prompts = du.build_prompts(train_ds, config['data']['data_path'])
    val_prompts = du.build_prompts(val_ds, config['data']['data_path'])
    train_targets = du.target_extraction(train_ds, config['data']['data_path'])
    val_targets = du.target_extraction(val_ds, config['data']['data_path'])
    reward_fn = lambda preds, tgts: du.compute_binary_reward(preds, tgts, config['data']['data_path'])

    # Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], betas=tuple(train_cfg['betas']))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=train_cfg.get('warmup', 100), num_training_steps=train_cfg['max_gen_len'])
    
    # Logging setup:
    t_board = SummaryWriter(log_dir=train_cfg['log_dir'])
    step_bar = tqdm(range(1, train_cfg['max_gen_len'] + 1), desc="train-step")

    # Training Loop:
    for step in step_bar:
        # Sampling N prompts:
        idx = random.sample(range(len(train_prompts)), train_cfg['batch_size'])
        prompts = [train_prompts[i] for i in idx]
        targets = [train_targets[i] for i in idx]

        # Forward-propagation:
        with context:
            loss, diag = gc.grpo_step(
                model, tokenizer, prompts, targets, reward_fn,
                num_generations=train_cfg['num_questions_per_batch'],
                max_new_tokens=train_cfg['max_gen_len'],
                device=config['model']['device'],
                ref_model=ref_model,
                kl_beta=train_cfg.get('kl_beta', 0.0),
                kl_epsilon=train_cfg.get('kl_epsilon', 0.2),
            )

        # Back-propagation:
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['max_grad_norm'])
            optimizer.step()
        scheduler.step()
        
        # Logging process:
        step_bar.set_postfix(loss=float(loss), reward=diag["raw_reward_mean"])
        t_board.add_scalar("train/loss", float(loss), step)
        t_board.add_scalar("train/r_mean", diag["raw_reward_mean"], step)
        t_board.add_scalar("train/kl_beta", train_cfg.get('kl_beta', 0.0), step)
        t_board.add_scalar("train/kl_epsilon", train_cfg.get('kl_epsilon', 0.2), step)
        t_board.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], step)
        
        # Evaluation and checkpointing:
        if step % train_cfg['eval_interval'] == 0 or step == train_cfg['max_gen_len']:
            with context:
                acc = evaluation(model, tokenizer, val_prompts, val_targets, reward_fn, 
                              train_cfg['num_questions_per_batch'], 64, config['model']['device'])
            t_board.add_scalar("val/accuracy", acc, step)
            torch.save(model.state_dict(), f"{train_cfg['ckpt_dir']}/model_step_{step}.pt")
            model.train()
        
    # Close logging:
    t_board.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
