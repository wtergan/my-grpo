# ===============================================================================
# IMPORTS AND MODEL SETUP
# ===============================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_core import generate_completions, group_rewards_normalization, token_policy_loss, grpo_step

# ===============================================================================
# MODEL SETUP
# ===============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# ===============================================================================
# GENERATION AND SHAPE TESTS
# ===============================================================================
def test_generate_completions_shapes():
    prompts = ["hello", "world"]
    batch = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu")
    assert batch["input_ids"].shape[0] == 4  # N*M
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    print("generate_completions shapes OK")

# ===============================================================================
# ADVANTAGE NORMALIZATION TESTS
# ===============================================================================
def test_group_rewards_normalization():
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    print("✅ group_rewards_normalization zero mean/unit std OK")

# ===============================================================================
# POLICY LOSS TESTS (PG & KL)
# ===============================================================================
# Test policy loss requires gradient:
def test_token_policy_loss_requires_grad():
    prompts = ["foo", "bar"]
    batch = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu")
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    loss = token_policy_loss(model, batch, adv)
    assert loss.requires_grad
    print("token_policy_loss requires_grad OK")

# Test policy loss PG vs KL:
def test_token_policy_loss_pg_vs_kl():
    prompts = ["foo", "bar"]
    batch_pg = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu", compute_old_log_probs=False)
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    # Simple PG loss
    loss_pg = token_policy_loss(model, batch_pg, adv)
    assert loss_pg.requires_grad
    # KL loss (PPO-style)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    batch_kl = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu", compute_old_log_probs=True)
    loss_kl = token_policy_loss(
        model, batch_kl, adv,
        old_log_probs=batch_kl["old_log_probs"],
        ref_model=ref_model,
        kl_beta=0.5,
        kl_epsilon=0.2
    )
    assert loss_kl.requires_grad
    assert loss_kl != loss_pg
    print("token_policy_loss PG vs KL OK")

# Test policy loss raises on missing old_log_probs:
def test_token_policy_loss_raises_on_missing_old_log_probs():
    prompts = ["foo", "bar"]
    batch = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu", compute_old_log_probs=False)
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    try:
        # Should fall back to PG loss, not raise
        token_policy_loss(model, batch, adv, ref_model=ref_model, kl_beta=0.5)
    except Exception as e:
        assert False, f"Should not raise when old_log_probs missing, got: {e}"
    # Now forcibly pass old_log_probs with wrong shape
    batch_bad = batch.copy()
    batch_bad["old_log_probs"] = torch.randn(1, 1)  # wrong shape
    try:
        token_policy_loss(model, batch_bad, adv, old_log_probs=batch_bad["old_log_probs"], ref_model=ref_model, kl_beta=0.5)
    except Exception as e:
        print("Caught expected error for shape mismatch:", e)
    else:
        assert False, "Should raise on old_log_probs shape mismatch"

# Test policy loss shape mismatch:
def test_token_policy_loss_shape_mismatch():
    prompts = ["foo", "bar"]
    batch = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu", compute_old_log_probs=True)
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Purposely pass advantages of wrong shape
    adv_bad = torch.tensor([1.])
    try:
        token_policy_loss(model, batch, adv_bad, old_log_probs=batch["old_log_probs"], ref_model=ref_model, kl_beta=0.5)
    except Exception as e:
        print("Caught expected error for advantages shape mismatch:", e)
    else:
        assert False, "Should raise on advantages shape mismatch"

# ===============================================================================
# GRPO STEP TESTS (PG & KL)
# ===============================================================================
# Test basic GRPO step:
def test_grpo_step_basic():
    prompts = ["a+b=", "c+d="]
    targets = ["2", "3"]
    def reward_fn(preds, tgts):
        # Dummy reward: +1 if target digit appears in prediction, else 0
        return torch.tensor([float(t in p) for p, t in zip(preds, tgts)])
    loss, diagnostics = grpo_step(
        model, tok, prompts, targets, reward_fn,
        num_generations=2, max_new_tokens=2, device="cpu"
    )
    assert isinstance(loss, torch.Tensor)
    assert "raw_reward_mean" in diagnostics
    assert "adv_std" in diagnostics
    print("grpo_step basic OK")

# Test PG vs KL:
def test_grpo_step_pg_vs_kl():
    prompts = ["a+b=", "c+d="]
    targets = ["2", "3"]
    def reward_fn(preds, tgts):
        return torch.tensor([float(t in p) for p, t in zip(preds, tgts)])
    # Simple PG
    loss_pg, diagnostics_pg = grpo_step(
        model, tok, prompts, targets, reward_fn,
        num_generations=2, max_new_tokens=2, device="cpu",
        ref_model=None, kl_beta=0.0
    )
    # KL/GRPO
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Force ref_model to be different from model by altering its weights
    with torch.no_grad():
        for param in ref_model.parameters():
            # Add small perturbations to the ref_model weights, for testing.
            param.add_(torch.randn_like(param) * 1e-4)  
    loss_kl, diagnostics_kl = grpo_step(
        model, tok, prompts, targets, reward_fn,
        num_generations=2, max_new_tokens=2, device="cpu",
        ref_model=ref_model, kl_beta=0.5, kl_epsilon=0.2
    )
    assert isinstance(loss_pg, torch.Tensor)
    assert isinstance(loss_kl, torch.Tensor)
    print(f"PG loss: {loss_pg.item()}, KL loss: {loss_kl.item()}")
    print("grpo_step PG vs KL OK")

# ===============================================================================
# MAIN TEST ENTRY POINT
# ===============================================================================
if __name__ == "__main__":
    test_generate_completions_shapes()
    test_group_rewards_normalization()
    test_token_policy_loss_requires_grad()
    test_token_policy_loss_pg_vs_kl()
    test_token_policy_loss_raises_on_missing_old_log_probs()
    test_token_policy_loss_shape_mismatch()
    test_grpo_step_basic()
    test_grpo_step_pg_vs_kl()
    print("All grpo_core.py tests passed!")
