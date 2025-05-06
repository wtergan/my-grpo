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

if __name__ == "__main__":
    show_batch()