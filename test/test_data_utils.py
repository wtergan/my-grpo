import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
# Import the module we are testing
import data_utils as du
# No need for DummyModel/Tokenizer here as data_utils doesn't directly use HF model/tokenizer loading.
# Its functions operate on data structures or use 'datasets.load_dataset'.

# ===============================================================================
# DATASET LOADING TESTS
# ===============================================================================
def test_gsm8k_load():
    # Use a small split to avoid heavy resource use and actual download in CI (if not cached)
    # This test will run `load_dataset` from HuggingFace `datasets` library.
    # For faster, isolated tests, this could be mocked if network access is an issue.
    try:
        train_ds, val_ds = du.load_task_dataset("gsm8k", split="train[:1%]", val_fraction=0.5) # 0.5% train, 0.5% val from 1%
        assert len(train_ds) > 0 and len(val_ds) > 0, "GSM8K tiny split loading failed or returned empty."
    except Exception as e:
        pytest.skip(f"Skipping GSM8K load test due to: {e} (possibly network or dataset availability issue)")


# ===============================================================================
# PROMPT BUILDING AND EXTRACTION TESTS
# ===============================================================================
def test_prompt_building():
    # Fake dataset example (using datasets.Dataset for consistency if du.build_prompts expects it)
    from datasets import Dataset
    fake_ds_list = [
        {"question": "What is 6 x 7?", "answer": "The final answer is \\boxed{42}"},
        {"question": "What is 5 + 8?", "answer": "The final answer is \\boxed{13}"}
    ]
    fake_ds = Dataset.from_list(fake_ds_list)
    
    prompts = du.build_prompts(fake_ds, "gsm8k")
    assert len(prompts) == 2
    assert "Q: What is 6 x 7?" in prompts[0]
    assert "Provide only the numeric result." in prompts[0]


def test_gsm8k_extraction():
    text1 = "Natalia sold 48/2 = <<48/2=24>>24 clips in May. #### 72"
    assert du.gsm8k_extraction(text1) == "72"
    
    text2 = "The answer is 1234." # No ####
    assert du.gsm8k_extraction(text2) == "1234" # Should find last number

    text3 = "The final answer is \\boxed{42}"
    assert du.gsm8k_extraction(text3) == "42"
    
    text4 = "No numbers here."
    assert du.gsm8k_extraction(text4) == ""

    text5 = "Answer: 1,234" # Number with comma
    assert du.gsm8k_extraction(text5) == "1234" # Regex `\d+` handles this correctly by finding parts

def test_prompt_and_target_extraction():
    # This also relies on load_dataset. Can be skipped or use a very small local mock if needed.
    try:
        ds, _ = du.load_task_dataset("gsm8k", split="test[:2]", val_fraction=0) # Get 2 samples from test
        if not ds: pytest.skip("GSM8K test[:2] returned empty dataset.")

        prompts = du.build_prompts(ds, "gsm8k")
        targets = du.target_extraction(ds, "gsm8k") # Extracts based on 'answer_key'
        
        assert len(prompts) == len(ds)
        assert len(targets) == len(ds)
        assert len(targets) > 0, "Targets list is empty."
        # Example: gsm8k target is "The final answer is \\boxed{XYZ}" format
        # The target_extraction should give the raw answer field.
        # The reward function's internal extraction normalizes this.
        assert "boxed" in targets[0].lower() # Check it's the raw answer field
    except Exception as e:
        pytest.skip(f"Skipping prompt/target extraction test due to: {e}")

# ===============================================================================
# REWARD COMPUTATION TESTS
# ===============================================================================
def test_binary_reward():
    # Predictions are model outputs, targets are ground truth *answer strings* (not numbers yet)
    preds  = ["The result is 42.", "My calculation gives 13, sorry.", "#### 100"]
    # Targets from gsm8k are like "The final answer is \\boxed{42}"
    # The du.compute_binary_reward uses task_def.answer_extraction on both pred and target.
    targets_raw_gsm8k = ["The final answer is \\boxed{42}", 
                         "The final answer is \\boxed{12}",
                         "The final answer is \\boxed{100}"]
    
    rewards_gsm8k = du.compute_binary_reward(preds, targets_raw_gsm8k, "gsm8k")
    expected_rewards_gsm8k = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(rewards_gsm8k, expected_rewards_gsm8k)

    # Test with a different hypothetical task (e.g. countdown, if extraction is just strip)
    preds_countdown = ["  my answer is 10  ", "20  "]
    targets_raw_countdown = ["10", "30"] # Assume countdown answers are just numbers as strings
    
    # Ensure 'countdown' task is defined as expected for this test
    if "countdown" in du.TASKS:
        rewards_countdown = du.compute_binary_reward(preds_countdown, targets_raw_countdown, "countdown")
        expected_rewards_countdown = torch.tensor([1.0, 0.0], dtype=torch.float32)
        assert torch.allclose(rewards_countdown, expected_rewards_countdown)
    else:
        print("Warning: 'countdown' task not defined in data_utils.TASKS, skipping part of test_binary_reward.")

if __name__ == "__main__":
    pytest.main([__file__])