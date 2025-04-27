"""
Core logic for GRPO: Batch generation, reward and advantage computation, policy loss.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import math
import torch
from torch.nn import functional as F

# ================================================================================
# 1. Batch Generation
# ================================================================================

@torch.inference_mode()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    num_generations: int = 4,
    max_new_tokens: int = 64,
    device: torch.device | str = "cuda",
    **gen_kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Batches and repeats prompts, 'num_generations' times, tokenizes them, then generates
    completions using given model.
    Returns a dict with: 
        - input_ids (batch, seq_len_prompt+completion).
        - attention_mask (batch, seq_len_prompt+completion) of ones.
        - prompt_lens (len(prompt_i) for masking later).
        - num_generations.
    """
    # Batch and repeat prompts:
    repeated_prompts: List[str] = sum([[p] * num_generations for p in prompts], [])
    # Tokenize the list of repeated prompts:
    enc = tokenizer(
        repeated_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,        
    ).to(device)
    # Generate completions:
    gen_tokens = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    # Output preparation: 
    prompt_length = enc["input_ids"].shape[1]
    full_ids = gen_tokens
    attn_mask = torch.ones_like(full_ids, dtype=torch.long)
    return dict(
        input_ids=full_ids,
        attention_mask=attn_mask,
        prompt_len=prompt_length,
        num_generations=num_generations,
    )

# ================================================================================
# 2. Reward Normalization
# ================================================================================

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
    stds = rewards.std(dim=1, keepdim=True).clamp_min(eps)
    advantages = (rewards - means) / stds
    return advantages.view(-1)

# ================================================================================
# 3. Token-level Policy Loss
# ================================================================================

def token_policy_loss(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    Takes the batch of generated completions and computed advantages, and returns
    the token-level policy gradient loss.
    batch: dict of (input_ids, attention_mask, prompt_len, num_generations)
    advantages: (N*group_size)
    scalar loss tensor (requires_grad)
    """
    input_ids = batch["input_ids"]            # (batch, seq_len)
    attn_mask = batch["attention_mask"]       # (batch, seq_len)
    prompt_len = batch["prompt_len"]          # int
    num_generations = batch["num_generations"]# int
    device = input_ids.device
    advantages = advantages.to(device)         # (batch,)

    # Forward pass to get the logits:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )
    logits = outputs.logits                   # (batch, seq_len, vocab_size)

    # Computation of the log-probabilities using logits (log_softmax):
    log_probs = F.log_softmax(logits, dim=-1) # (batch, seq_len, vocab_size)

    # Select log-probs for completion tokens (teacher forcing):
    # Start at prompt_len-1 (last prompt token predicts first completion token), stop at -1.
    comp_log_probs = log_probs[:, prompt_len - 1 : -1, :]   # (batch, completion_len, vocab_size)
    comp_targets = input_ids[:, prompt_len:]                 # (batch, completion_len)
    # Gather log-probabilities for the target tokens:
    token_logp = comp_log_probs.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)  # (batch, completion_len)
    # Mask padding tokens:
    comp_mask = (comp_targets != model.config.pad_token_id).float()                 # (batch, completion_len)
    # Broadcast advantages:
    adv_broadcast = advantages.unsqueeze(1) * comp_mask     # (batch, completion_len)
    # Compute the loss:
    loss = -adv_broadcast * token_logp                     # (batch, completion_len)
    return loss.sum() / comp_mask.sum().clamp_min(1)       # scalar

# ================================================================================
# 4. GRPO Step
# ================================================================================

def grpo_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    targets: List[str],
    reward_fn,
    num_generations: int = 4,
    max_new_tokens: int = 64,
    device: torch.device | str = "cuda",
) -> Tuple[torch.Tensor, Dict]:
    """
    Brings everything together: generation, reward, and policy loss.
    Returns (loss, diagnostics)
    """
    batch = generate_completions(
        model, tokenizer, prompts, 
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    # Decode predictions:
    pred_texts = tokenizer.batch_decode(batch["input_ids"][:, batch["prompt_len"] :], skip_special_tokens=True)
    rewards = reward_fn(pred_texts, targets * num_generations)  # Targets repeated.
    advantages = group_rewards_normalization(rewards, num_generations)
    loss = token_policy_loss(model, batch, advantages)
    diagnostics = {
        "raw_reward_mean": rewards.mean().item(),
        "adv_std": advantages.std().item(),
    }
    return loss, diagnostics
