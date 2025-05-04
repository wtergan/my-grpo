"""
Core logic for GRPO: Batch generation, reward and advantage computation, policy loss.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import math
import torch
from torch.nn import functional as F

# ===============================================================================
# BATCH GENERATION
# ===============================================================================
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    num_generations: int = 4,
    max_new_tokens: int = 64,
    device: torch.device | str = "cuda",
    compute_old_log_probs: bool = False,
    **gen_kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Batches and repeats prompts, 'num_generations' times, tokenizes them, then generates
    completions using given model.
    Returns a dict with: 
        - input_ids (batch, seq_len_prompt+completion).
        - attention_mask (batch, seq_len_prompt+completion) of ones.
        - prompt_len (len(prompt_i) for masking later).
        - num_generations.
        - old_log_probs (log-probs for completions under current model, if requested).
    """
    repeated_prompts: List[str] = sum([[p] * num_generations for p in prompts], [])
    enc = tokenizer(
        repeated_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,        
    ).to(device)
    gen_tokens = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    prompt_length = enc["input_ids"].shape[1]
    full_ids = gen_tokens
    attn_mask = torch.ones_like(full_ids, dtype=torch.long)
    batch = dict(
        input_ids=full_ids,
        attention_mask=attn_mask,
        prompt_len=prompt_length,
        num_generations=num_generations,
    )
    if compute_old_log_probs:
        batch["old_log_probs"] = compute_log_probs(model, full_ids, attn_mask, prompt_length)
    return batch

def compute_log_probs(model, input_ids, attention_mask, prompt_len):
    """Returns log-probs for completions (tokens after prompt) for each sequence in batch."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        comp_targets = input_ids[:, prompt_len:]
        comp_log_probs = log_probs[:, prompt_len - 1 : -1, :] # TEACHER FORCING
        old_log_probs = comp_log_probs.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)
    return old_log_probs

# ===============================================================================
# REWARD NORMALIZATION
# ===============================================================================

def group_rewards_normalization(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Takes rewards (N*group_size) and normalizes them within each group.
    Returns the advantages (N*group_size) (zero mean, unit variance within each group).
    """
    if rewards.ndim != 1:
        rewards = rewards.flatten()
    assert rewards.numel() % group_size == 0, "rewards length not divisible by group_size"
    n_prompts = rewards.numel() // group_size
    rewards = rewards.view(n_prompts, group_size)
    means = rewards.mean(dim=1, keepdim=True)
    # Use unbiased=False for unit variance (uses n for population, not n-1 for sample).
    stds = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    advantages = (rewards - means) / stds
    return advantages.view(-1)

# ===============================================================================
# TOKEN-LEVEL POLICY LOSS: Simple PG OR PG w/KL Divergence+Reference.
# ===============================================================================

def token_policy_loss(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor = None,  
    ref_model=None,
    kl_beta=0.0,
    kl_epsilon=0.2,  
) -> torch.Tensor:
    """
    Computes the token-level policy gradient loss.
    Supports both simple PG and PPO-style GRPO with KL+reference.
    Args:
        model: The current model.
        batch: dict with keys input_ids, attention_mask, prompt_len.
        advantages: (N*group_size).
        old_log_probs: The log-probs from the model at rollout (for PPO ratio).
        ref_model: Reference model for KL penalty.
        kl_beta: KL penalty weight.
        kl_epsilon: PPO clipping epsilon.
    """
    input_ids = batch["input_ids"]            # (batch, seq_len)
    attn_mask = batch["attention_mask"]       # (batch, seq_len)
    prompt_len = batch["prompt_len"]          # int
    device = input_ids.device
    advantages = advantages.to(device)

    # Forward pass to get the logits (current model):
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits                   # (batch, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1) # (batch, seq_len, vocab_size)
    comp_log_probs = log_probs[:, prompt_len - 1 : -1, :]   # (batch, completion_len, vocab_size) TEACHER FORCING.
    comp_targets = input_ids[:, prompt_len:]                 # (batch, completion_len)
    new_log_probs = comp_log_probs.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)  # (batch, completion_len)
    # Padding-mask:
    pad_id = (model.config.pad_token_id if model.config.pad_token_id is not None else 
              getattr(model.config, "eos_token_id", None))
    if pad_id is None:
        comp_mask = torch.ones_like(comp_targets, dtype=torch.float)
    else:
        comp_mask = comp_targets.ne(pad_id).float()         # (batch, completion_len)

    # Shape checks:
    # Check advantages shape
    if advantages.shape[0] != input_ids.shape[0]:
        raise ValueError(f"advantages shape {advantages.shape} does not match batch size {input_ids.shape[0]}")
    # Check old_log_probs shape if provided:
    if kl_beta != 0.0 and ref_model is not None and old_log_probs is not None:
        if old_log_probs.shape != new_log_probs.shape:
            raise ValueError(f"old_log_probs shape {old_log_probs.shape} does not match new_log_probs shape {new_log_probs.shape}")

    adv_broadcast = advantages.unsqueeze(1) * comp_mask     # (batch, completion_len)

    # ==================== SIMPLE POLICY GRADIENT LOSS ====================
    if kl_beta == 0.0 or ref_model is None or old_log_probs is None:
        pg_loss = -adv_broadcast * new_log_probs           # (batch, completion_len)
        loss = pg_loss.sum() / comp_mask.sum().clamp_min(1)
        return loss

    # ==================== KL + REFERENCE LOSS (PPO-style) ====================
    # old_log_probs should be provided from rollout (model before update).
    # Compute reference log-probs for KL penalty:
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask=attn_mask).logits
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_comp_logp = ref_log_probs[:, prompt_len - 1 : -1, :].gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)

    # PPO ratio
    ratio = torch.exp(new_log_probs - old_log_probs)  # (batch, completion_len)
    unclipped = ratio * advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1 - kl_epsilon, 1 + kl_epsilon) * advantages.unsqueeze(1)
    surrogate_loss = torch.min(unclipped, clipped) * comp_mask

    # KL term (per-token, reverse KL)
    kl = torch.exp(ref_comp_logp - new_log_probs) - (ref_comp_logp - new_log_probs) - 1  # (batch, completion_len)
    per_token_loss = surrogate_loss - kl_beta * kl * comp_mask
    # Final loss (mean over non-pad tokens per sample, then mean over batch):
    per_sample_loss = (per_token_loss * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp_min(1)
    loss = -per_sample_loss.mean()
    return loss

# ===============================================================================
# GRPO STEP
# ===============================================================================

def grpo_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    targets: List[str],
    reward_fn,
    num_generations: int = 4,
    max_new_tokens: int = 64,
    device: torch.device | str = "cuda",
    ref_model=None,
    kl_beta: float = 0.0,
    kl_epsilon: float = 0.2,
) -> Tuple[torch.Tensor, Dict]:
    """
    Brings everything together: generation, reward, and policy loss.
    Returns (loss, diagnostics).
    """
    compute_pg = (kl_beta == 0.0 or ref_model is None)
    batch = generate_completions(
        model, tokenizer, prompts, 
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        device=device,
        compute_old_log_probs=not compute_pg,
    )
    pred_texts = tokenizer.batch_decode(batch["input_ids"][:, batch["prompt_len"] :], skip_special_tokens=True)
    rewards = reward_fn(pred_texts, targets * num_generations)
    advantages = group_rewards_normalization(rewards, num_generations)
    loss = token_policy_loss(
        model,
        batch,
        advantages,
        old_log_probs=batch.get("old_log_probs", None),
        ref_model=ref_model,
        kl_beta=kl_beta,
        kl_epsilon=kl_epsilon,
    )
    diagnostics = {
        "raw_reward_mean": rewards.mean().item(),
        "adv_std": advantages.std().item(),
    }
    return loss, diagnostics
