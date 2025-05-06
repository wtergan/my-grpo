import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_core import generate_completions

def show_batch():
    # Load a small model for demonstration
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Example prompts
    prompts = ["Hello world!", "How are you?"]

    # Generate completions
    batch = generate_completions(
        model,
        tokenizer,
        prompts,
        num_generations=2,
        max_new_tokens=4,
        device=device
    )

    # Show anatomy of the batch
    print("Returned batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("input_ids tensor:\n", batch["input_ids"])
    print("Decoded sequences:")
    for i, ids in enumerate(batch["input_ids"]):
        print(f"Sample {i}: {tokenizer.decode(ids)}")
    print("prompt_len:", batch["prompt_len"])
    print("num_generations:", batch["num_generations"])
    if "old_log_probs" in batch:
        print("old_log_probs shape:", batch["old_log_probs"].shape)

    # Full softmax log-prob computation (no chunking)
    import torch.nn.functional as F
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    prompt_len = batch["prompt_len"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)
        print("logits shape:", logits.shape)
        print("logits sample:", logits[0, :prompt_len+2, :8])
        comp_logits = logits[:, prompt_len - 1 : -1, :]
        print("comp_logits shape:", comp_logits.shape)
        print("comp_logits sample:", comp_logits[0, :2, :8])
        comp_targets = input_ids[:, prompt_len:]
        print("comp_targets shape:", comp_targets.shape)
        print("comp_targets sample:", comp_targets[0, :8])
        # Full softmax over all completions at once
        log_probs_full = F.log_softmax(comp_logits, dim=-1)
        print("log_probs_full shape:", log_probs_full.shape)
        print("log_probs_full sample:", log_probs_full[0, :2, :8])
        gathered = log_probs_full.gather(-1, comp_targets.unsqueeze(-1)).squeeze(-1)
        print("gathered log_probs shape:", gathered.shape)
        print("gathered log_probs sample:", gathered[0, :8])
        log_probs = gathered
        print("Final log_probs shape:", log_probs.shape)
        print("Final log_probs sample:", log_probs[0, :8])

if __name__ == "__main__":
    show_batch()