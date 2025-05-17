import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F # For direct comparison in compute_log_probs test

# Import functions to test from grpo_core
from grpo_core import (generate_completions, group_rewards_normalization, 
                       token_policy_loss, grpo_step, compute_log_probs)

# Transformers imports for type patching, FakeAutoX will be used via conftest.py
import transformers
# Dummy classes will be loaded from conftest.py by pytest

MODEL_NAME_PLACEHOLDER = "dummy/model" # Placeholder, actual model not loaded

@pytest.fixture(autouse=True)
def patch_hf_imports(monkeypatch, request):
    # This fixture applies to all tests in this file.
    # It patches AutoTokenizer and AutoModelForCausalLM within the 'transformers' module.
    # This is because grpo_core itself doesn't import them, but tests might instantiate them
    # or they might be passed from train.py (which would use patched versions).
    # The FakeAutoX classes are from conftest.py.
    # We need to ensure `request.getfixturevalue` can access them from conftest.
    FakeAutoTokenizer = request.getfixturevalue('FakeAutoTokenizer')
    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    
    monkeypatch.setattr(transformers, 'AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr(transformers, 'AutoModelForCausalLM', FakeAutoModel)

@pytest.fixture
def dummy_model_and_tokenizer(request):
    # Get the fake classes defined in conftest.py
    FakeAutoTokenizer = request.getfixturevalue('FakeAutoTokenizer')
    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    
    tokenizer = FakeAutoTokenizer.from_pretrained(MODEL_NAME_PLACEHOLDER)
    model = FakeAutoModel.from_pretrained(MODEL_NAME_PLACEHOLDER)
    model.eval() # Default to eval mode for tests not involving training
    return model, tokenizer

# ===============================================================================
# GENERATION AND SHAPE TESTS
# ===============================================================================
def test_generate_completions_shapes(dummy_model_and_tokenizer):
    model, tokenizer = dummy_model_and_tokenizer
    prompts = ["hello", "world example"]
    batch = generate_completions(model, tokenizer, prompts, num_generations=2, max_new_tokens=3, device="cpu")
    
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "prompt_len" in batch
    assert "num_generations" in batch

    # N prompts * M generations
    expected_batch_size = len(prompts) * 2 
    assert batch["input_ids"].shape[0] == expected_batch_size
    assert batch["attention_mask"].shape[0] == expected_batch_size
    
    # Seq len = prompt_len + max_new_tokens
    # Dummy tokenizer might give fixed prompt_len, or vary. Check based on actual output.
    # tokenizer("hello", return_tensors="pt").input_ids.shape[1] would be actual prompt len.
    # For the dummy tokenizer, let's assume its __call__ pads to a certain length or uses text length.
    # The 'prompt_len' in batch is the length of the tokenized prompts *before* generation.
    # The total length is batch['prompt_len'] + max_new_tokens (for the longest sequence).
    # The dummy generate pads new tokens to max_new_tokens.
    # The dummy tokenizer pads input prompts to the max length within the repeated_prompts batch.
    # Let's verify based on the returned prompt_len.
    expected_seq_len = batch["prompt_len"] + 3 # 3 is max_new_tokens
    assert batch["input_ids"].shape[1] == expected_seq_len
    assert batch["attention_mask"].shape[1] == expected_seq_len
    assert batch["num_generations"] == 2


# ===============================================================================
# ADVANTAGE NORMALIZATION TESTS
# ===============================================================================
def test_group_rewards_normalization():
    rewards1 = torch.tensor([1., 0., 0., 1.]) # 2 groups of 2
    adv1 = group_rewards_normalization(rewards1, group_size=2)
    assert torch.isclose(adv1.mean(), torch.tensor(0.0), atol=1e-6)
    # For [1,0], mean=0.5, std=0.5. (1-0.5)/0.5=1. (0-0.5)/0.5=-1. So adv for first group is [1,-1]
    # For [0,1], mean=0.5, std=0.5. (0-0.5)/0.5=-1. (1-0.5)/0.5=1. So adv for second group is [-1,1]
    # Combined std is not 1. It's std within each group that is 1 (if not all same).
    # Check std within groups:
    adv1_reshaped = adv1.view(-1, 2)
    assert torch.isclose(adv1_reshaped[0].std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(adv1_reshaped[1].std(unbiased=False), torch.tensor(1.0), atol=1e-6)

    rewards2 = torch.tensor([10.0, 10.0, 10.0]) # 1 group of 3, all same
    adv2 = group_rewards_normalization(rewards2, group_size=3)
    # If all rewards in a group are same, std is 0. Advantages should be 0 to avoid div by zero.
    assert torch.allclose(adv2, torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)

    rewards3 = torch.tensor([1., 2., 3., 4., 5., 6.]) # 2 groups of 3
    adv3 = group_rewards_normalization(rewards3, group_size=3)
    assert torch.isclose(adv3.mean(), torch.tensor(0.0), atol=1e-6)
    adv3_reshaped = adv3.view(-1, 3)
    assert torch.isclose(adv3_reshaped[0].std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(adv3_reshaped[1].std(unbiased=False), torch.tensor(1.0), atol=1e-6)


# ===============================================================================
# POLICY LOSS TESTS (PG & KL)
# ===============================================================================
def test_token_policy_loss_requires_grad(dummy_model_and_tokenizer):
    model, tokenizer = dummy_model_and_tokenizer
    model.train() # Ensure model is in train mode for gradients
    
    prompts = ["foo", "bar"]
    # compute_old_log_probs=True because token_policy_loss might use old_log_probs
    # even if ref_model is None (for PPO ratio, though it should fall back to PG if ref_model is None or kl_beta=0)
    # Let's test simple PG case first, kl_beta=0, no ref_model, no old_log_probs needed by logic.
    batch = generate_completions(model, tokenizer, prompts, num_generations=1, max_new_tokens=2, device="cpu", compute_old_log_probs=False)
    rewards = torch.tensor([1., 0.]) 
    adv = group_rewards_normalization(rewards, group_size=1) # Effectively just normalizes if N>1

    loss = token_policy_loss(model, batch, adv, kl_beta=0.0) # Simple PG.
    assert loss.requires_grad, "PG loss should require gradients."
    loss.backward() # Check if backward pass runs
    assert model.dummy_param.grad is not None, "Gradients not flowing to model parameters for PG loss."

def test_token_policy_loss_pg_vs_kl(dummy_model_and_tokenizer, request):
    model, tokenizer = dummy_model_and_tokenizer
    model.train()
    model.dummy_param.grad = None # Reset grad

    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    ref_model = FakeAutoModel.from_pretrained("dummy/ref_model")
    ref_model.to("cpu").eval()

    prompts = ["foo query", "bar query"]
    num_gens = 2
    max_tokens = 3
    
    # Generate batch with old_log_probs for KL case, since we need these old logprobs for the KL-term.
    batch_data = generate_completions(model, tokenizer, prompts, 
                                      num_generations=num_gens, max_new_tokens=max_tokens, 
                                      device="cpu", compute_old_log_probs=True)
    
    rewards = torch.rand(len(prompts) * num_gens) # Random rewards
    advantages = group_rewards_normalization(rewards, group_size=num_gens)

    # 1. Simple PG loss (kl_beta = 0).
    loss_pg = token_policy_loss(
        model, batch_data, advantages.clone(), # .clone() so as to not mutate original adv, but keep autograd history.
        old_log_probs=None, # PG doesn't use old_log_probs from batch.
        ref_model=None, 
        kl_beta=0.0
    )
    assert loss_pg.requires_grad
    loss_pg.backward()
    pg_grad_norm = model.dummy_param.grad.norm().item()
    model.dummy_param.grad = None # Reset grad

    # 2. KL loss (PPO-style), in which reference model and probs, old logprobs and kl-beta/eps are needed.
    assert "old_log_probs" in batch_data, "old_log_probs missing for KL test."
    loss_kl = token_policy_loss(
        model, batch_data, advantages.clone(),
        old_log_probs=batch_data["old_log_probs"],
        ref_model=ref_model,
        kl_beta=0.5,
        kl_epsilon=0.2
    )
    assert loss_kl.requires_grad
    loss_kl.backward()
    kl_grad_norm = model.dummy_param.grad.norm().item()
    model.dummy_param.grad = None # Reset grad
    
    # Losses should be different in general, and grads too.
    assert not torch.isclose(loss_pg, loss_kl), "PG and KL losses should ideally be different for this setup."
    assert pg_grad_norm != kl_grad_norm or pg_grad_norm == 0, "PG and KL grad norms imply different updates or no update."


def test_token_policy_loss_fallback_and_errors(dummy_model_and_tokenizer, request):
    model, tokenizer = dummy_model_and_tokenizer
    model.train()
    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    ref_model = FakeAutoModel.from_pretrained("dummy/ref_model").to("cpu").eval()

    prompts = ["test prompt"]
    batch = generate_completions(model, tokenizer, prompts, num_generations=1, max_new_tokens=2, device="cpu", compute_old_log_probs=False) # No old_log_probs
    adv = torch.tensor([1.0])

    # Case 1: kl_beta > 0, ref_model present, but old_log_probs missing from batch.
    # Should fall back to PG loss as per current logic in token_policy_loss.
    try:
        loss_fallback = token_policy_loss(model, batch, adv, ref_model=ref_model, kl_beta=0.5)
        assert loss_fallback is not None # Check it ran
    except Exception as e:
        assert False, f"Should fall back to PG loss, but raised: {e}"

    # Case 2: old_log_probs provided but with wrong shape (for KL part). 
    batch_with_old_log_probs = generate_completions(model, tokenizer, prompts, num_generations=1, max_new_tokens=2, device="cpu", compute_old_log_probs=True)
    correct_old_log_probs = batch_with_old_log_probs["old_log_probs"]
    
    # Create bad old_log_probs.
    bad_old_log_probs = torch.randn(correct_old_log_probs.shape[0] + 1, correct_old_log_probs.shape[1]) 
    with pytest.raises(ValueError, match=r"old_log_probs shape.*does not match new_log_probs shape"):
        token_policy_loss(model, batch_with_old_log_probs, adv, old_log_probs=bad_old_log_probs, ref_model=ref_model, kl_beta=0.5)

    # Case 3: Advantages with wrong shape.
    bad_adv = torch.tensor([1.0, 2.0]) # Batch size is 1 for 'prompts'
    with pytest.raises(ValueError, match=r"advantages shape.*does not match batch size"):
         token_policy_loss(model, batch_with_old_log_probs, bad_adv, old_log_probs=correct_old_log_probs, ref_model=ref_model, kl_beta=0.5)


# ===============================================================================
# COMPUTE_LOG_PROBS TESTS
# ===============================================================================
def test_compute_log_probs_chunked_and_grad(dummy_model_and_tokenizer):
    model, tokenizer = dummy_model_and_tokenizer
    model.train() # For grad test

    prompts = ["hello example one", "world example two long"]
    batch = generate_completions(model, tokenizer, prompts, num_generations=1, max_new_tokens=8, device="cpu")
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    # Manually determine a prompt_len for testing compute_log_probs; it's an arg.
    # Let's say the tokenized "hello example one" becomes 3 tokens for the purpose of this test slice.
    # This must be less than input_ids.shape[1] - 1 to have some completion.
    test_prompt_len = min(3, input_ids.shape[1] - 2) 
    if test_prompt_len <=0: test_prompt_len = 1 # Ensure positive if seq too short

    # Test no_grad=True (for old log probs)
    log_probs_ng = compute_log_probs(model, input_ids, attn_mask, test_prompt_len, chunk_size=4, no_grad=True)
    assert log_probs_ng.shape[0] == input_ids.shape[0]
    assert log_probs_ng.shape[1] == input_ids.shape[1] - test_prompt_len # Completion length
    assert not log_probs_ng.requires_grad

    # Test no_grad=False (should require grad on output)
    log_probs_g = compute_log_probs(model, input_ids, attn_mask, test_prompt_len, chunk_size=4, no_grad=False)
    assert log_probs_g.requires_grad
    
    # Test backward works
    log_probs_g.sum().backward()
    assert model.dummy_param.grad is not None
    model.dummy_param.grad = None # Clear grad

    # Check chunking correctness (sum close to full softmax).
    # compute_log_probs with large chunk_size is effectively non-chunked.
    with torch.no_grad(): # Ensure this outer context for comparison
        log_probs_full_ref = compute_log_probs(model, input_ids, attn_mask, test_prompt_len, chunk_size=1000, no_grad=True)
    
    assert torch.allclose(log_probs_ng, log_probs_full_ref, atol=1e-5), "Chunked log_probs differ from non-chunked"


# ===============================================================================
# GRPO STEP TESTS (PG & KL)
# ===============================================================================
def test_grpo_step_basic(dummy_model_and_tokenizer):
    model, tokenizer = dummy_model_and_tokenizer
    model.train()

    prompts = ["a+b=", "c+d="]
    # Targets are raw strings, reward_fn will process them
    targets = ["solution for a+b", "solution for c+d"] 
    
    def dummy_reward_fn(predictions: list[str], true_targets: list[str]):
        # Dummy reward: +1 if target digit appears in prediction, else 0
        rewards = []
        for pred, target_str_raw in zip(predictions, true_targets):
            # Dummy: check if first char of target_str_raw is in pred
            rewards.append(float(target_str_raw[0] in pred))
        return torch.tensor(rewards, dtype=torch.float32)

    loss, diagnostics = grpo_step(
        model, tokenizer, prompts, targets, dummy_reward_fn,
        num_generations=2, max_new_tokens=2, device="cpu",
        kl_beta=0.0 # Basic PG
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert "raw_reward_mean" in diagnostics
    assert "adv_std" in diagnostics
    
    loss.backward()
    assert model.dummy_param.grad is not None
    model.dummy_param.grad = None


def test_grpo_step_pg_vs_kl(dummy_model_and_tokenizer, request):
    model, tokenizer = dummy_model_and_tokenizer
    model.train()
    
    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    ref_model = FakeAutoModel.from_pretrained("dummy/ref_model_for_grpo_step").to("cpu").eval()
    # Perturb ref_model slightly to ensure it's different from the main model.
    with torch.no_grad():
        for param in ref_model.parameters():
            param.add_(torch.randn_like(param) * 1e-3)

    prompts = ["query one", "query two"]
    targets = ["target one text", "target two text"]
    def dummy_reward_fn(preds, tgts): # Same dummy reward as above
        return torch.tensor([float(t[0] in p) for p, t in zip(preds, tgts)])

    # 1. Simple PG
    loss_pg, diag_pg = grpo_step(
        model, tokenizer, prompts, targets, dummy_reward_fn,
        num_generations=2, max_new_tokens=2, device="cpu",
        ref_model=None, kl_beta=0.0
    )
    loss_pg.backward()
    pg_grad_norm = model.dummy_param.grad.norm().item()
    model.dummy_param.grad = None

    # 2. KL/GRPO
    loss_kl, diag_kl = grpo_step(
        model, tokenizer, prompts, targets, dummy_reward_fn,
        num_generations=2, max_new_tokens=2, device="cpu",
        ref_model=ref_model, kl_beta=0.5, kl_epsilon=0.2
    )
    loss_kl.backward()
    kl_grad_norm = model.dummy_param.grad.norm().item()
    model.dummy_param.grad = None

    assert isinstance(loss_pg, torch.Tensor)
    assert isinstance(loss_kl, torch.Tensor)
    assert not torch.isclose(loss_pg, loss_kl), "PG and KL losses from grpo_step should differ"
    # Grad norms might be zero if updates are tiny or cancel out, but generally should differ if losses differ
    assert pg_grad_norm != kl_grad_norm or pg_grad_norm == 0.0 


if __name__ == "__main__":
    pytest.main([__file__])