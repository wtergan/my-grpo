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
# TEST: generate_completions
# ===============================================================================
def test_generate_completions_shapes():
    prompts = ["hello", "world"]
    batch = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu")
    assert batch["input_ids"].shape[0] == 4  # N*M
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    print("✅ generate_completions shapes OK")

# ===============================================================================
# TEST: group_rewards_normalization
# ===============================================================================
def test_group_rewards_normalization():
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    print("✅ group_rewards_normalization zero mean/unit std OK")

# ===============================================================================
# TEST: token_policy_loss
# ===============================================================================
def test_token_policy_loss_requires_grad():
    prompts = ["foo", "bar"]
    batch = generate_completions(model, tok, prompts, num_generations=2, max_new_tokens=2, device="cpu")
    rewards = torch.tensor([1., 0., 0., 1.])
    adv = group_rewards_normalization(rewards, 2)
    loss = token_policy_loss(model, batch, adv)
    assert loss.requires_grad
    print("✅ token_policy_loss requires_grad OK")

# ===============================================================================
# TEST: grpo_step
# ===============================================================================
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
    print("✅ grpo_step basic OK")

if __name__ == "__main__":
    test_generate_completions_shapes()
    test_group_rewards_normalization()
    test_token_policy_loss_requires_grad()
    test_grpo_step_basic()
    print("All grpo_core.py tests passed!")
