import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import data_utils as du
import torch

# Test dataset loading (only check function runs and returns non-empty for small split):
def test_gsm8k_load():
    # Use a small split to avoid heavy resource use:
    train, val = du.load_task_dataset("gsm8k", split="train", val_fraction=0.01)  # 1% val
    assert len(train) > 0 and len(val) > 0

# Test reward computation logic:ß
def test_binary_reward():
    preds  = ["42", "13"]
    target = ["42", "12"]
    r = du.compute_binary_reward(preds, target, "gsm8k")
    assert torch.allclose(r, torch.tensor([1.0, 0.0], dtype=torch.float32))

# Test prompt building:
def test_prompt_building():
    # Fake dataset example:
    fake_ds = du.Dataset.from_list([
        {"question": "What is 6 x 7?", "answer": "42"},
        {"question": "What is 5 + 8?", "answer": "13"}
    ])
    prompts = du.build_prompts(fake_ds, "gsm8k")
    assert all("Question:" in p or "Q:" in p for p in prompts)

# Test answer extraction logic:
def test_gsm8k_extraction():
    text = "Natalia sold 48/2 = <<48/2=24>>24 clips in May. #### 72"
    answer = du.gsm8k_extraction(text)
    assert answer == "72"

if __name__ == "__main__":
    pytest.main([__file__])
