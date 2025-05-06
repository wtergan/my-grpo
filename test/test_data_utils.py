# ===============================================================================
# IMPORTS AND SETUP
# ===============================================================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import data_utils as du
import torch

# ===============================================================================
# DATASET LOADING TESTS
# ===============================================================================
# Test dataset loading (only check function runs and returns non-empty for small split):
def test_gsm8k_load():
    # Use a small split to avoid heavy resource use:
    train, val = du.load_task_dataset("gsm8k", split="train", val_fraction=0.01)  # 1% val
    assert len(train) > 0 and len(val) > 0

# ===============================================================================
# PROMPT BUILDING AND EXTRACTION TESTS
# ===============================================================================
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

# Test prompt and target extraction:
def test_prompt_and_target_extraction():
    ds, _ = du.load_task_dataset("gsm8k", split="test[:1]")
    prompts = du.build_prompts(ds, "gsm8k")
    targets = du.target_extraction(ds, "gsm8k")
    assert len(prompts) == len(ds)
    assert len(targets) == len(ds)
    print("prompt and target extraction lengths OK")

# ===============================================================================
# REWARD COMPUTATION TESTS
# ===============================================================================
# Test reward computation logic:
def test_binary_reward():
    preds  = ["42", "13"]
    target = ["42", "12"]
    r = du.compute_binary_reward(preds, target, "gsm8k")
    assert torch.allclose(r, torch.tensor([1.0, 0.0], dtype=torch.float32))

# ===============================================================================
# PYTEST FIXTURES
# ===============================================================================
import pytest

class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = '[PAD]'
        self.eos_token_id = 0
    def add_special_tokens(self, *a, **kw): return None
    def batch_decode(self, *a, **kw): return ["dummy"]
    def __call__(self, *a, **kw):
        class Dummy:
            def to(self, device): return self
            input_ids = [[0]]
        return Dummy()

class DummyModel:
    config = type('config', (), {'pad_token_id': 0})()
    def resize_token_embeddings(self, n): return None
    def to(self, device): return self
    def eval(self): return self
    def generate(self, **kwargs): return [[0, 0, 0]]
    def __call__(self, *args, **kwargs):
        import torch
        class Dummy:
            def to(self, device): return self
        return Dummy()
    def parameters(self):
        return []

@pytest.fixture(autouse=True)
def patch_data_utils_dependencies(monkeypatch):
    try:
        import data_utils as du
        import transformers
        monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', staticmethod(lambda *a, **kw: DummyTokenizer()))
        monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', staticmethod(lambda *a, **kw: DummyModel()))
    except ImportError:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
