import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import yaml
import pytest
import tempfile # For robust temp dir creation if needed

# Module to test
import train
# Dependent modules that will be patched or whose functions will be patched
import data_utils 
import grpo_core # train.py calls gc.grpo_step

# Dummies/Fakes will be available from conftest.py

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.yaml')

def load_train_test_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config['training'], config['model'], config['data']

# This fixture will mock dependencies for all tests in this file.
@pytest.fixture(autouse=True)
def patch_train_module_dependencies(monkeypatch, request):
    FakeAutoTokenizer = request.getfixturevalue('FakeAutoTokenizer')
    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    DummyModel = request.getfixturevalue('DummyModel') # For ref_model type check perhaps
    DummyTokenizer = request.getfixturevalue('DummyTokenizer')


    # Patch Hugging Face model/tokenizer loading in train.py
    monkeypatch.setattr(train, 'AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr(train, 'AutoModelForCausalLM', FakeAutoModel)

    # Patch dataset loading from data_utils (which train.py imports as du)
    # Default mock data:
    default_train_ds = [{'question': 'TrainQ1', 'answer': 'TrainA1'}, {'question': 'TrainQ2', 'answer': 'TrainA2'}]
    default_val_ds = [{'question': 'ValQ1', 'answer': 'ValA1'}]
    monkeypatch.setattr(data_utils, 'load_task_dataset', 
                        lambda task_name, split="train", seed=42, val_fraction=0.1: (default_train_ds.copy(), default_val_ds.copy()))

    # Patch TensorBoard SummaryWriter to prevent disk writes
    class DummySummaryWriter:
        def __init__(self, log_dir=None): pass
        def add_scalar(self, tag, scalar_value, global_step=None): pass
        def close(self): pass
    monkeypatch.setattr(train, 'SummaryWriter', DummySummaryWriter)

    # Optional: Patch grpo_core.grpo_step if its internals are too complex or slow for train unit tests
    # For now, let it run with DummyModel as it tests integration.
    # def mock_grpo_step(*args, **kwargs):
    #     return torch.tensor(0.1, requires_grad=True), {"raw_reward_mean": 0.5, "adv_std": 1.0}
    # monkeypatch.setattr(grpo_core, 'grpo_step', mock_grpo_step)


# ===============================================================================
# SMOKE TESTS AND BASIC FUNCTIONALITY
# ===============================================================================
@pytest.mark.timeout(60) # Prevent indefinite hangs
def test_train_loop_cpu_smoke(tmp_path):
    train_cfg, model_cfg, data_cfg = load_train_test_config()

    # Override config for a quick test run
    model_cfg['device'] = 'cpu'
    model_cfg['dtype'] = 'float32' # Simpler for CPU testing with dummies
    train_cfg['steps'] = 3
    train_cfg['batch_n'] = 1 # Smallest batch
    train_cfg['gens_m'] = 1  # Smallest generations
    train_cfg['eval_every'] = 2
    train_cfg['save_dir'] = str(tmp_path / "test_checkpoints_smoke")
    # Ensure save_dir exists (train.py creates it, but good practice for tests using tmp_path)
    os.makedirs(train_cfg['save_dir'], exist_ok=True) 
    
    train.main(train_cfg, model_cfg, data_cfg) # Execute the main training loop

    # Check if a checkpoint file was created (as per eval_every or final step)
    # Step 3 is > eval_every=2, so step 2 should save. Final step also saves.
    expected_ckpt_path = os.path.join(train_cfg['save_dir'], f"model_step_{train_cfg['steps']}.pt")
    assert os.path.exists(expected_ckpt_path), "Checkpoint file not created at the end of training."


# ===============================================================================
# EDGE CASES: EMPTY AND SINGLE SAMPLE DATASETS
# ===============================================================================
def test_train_with_empty_dataset_should_fail(monkeypatch, tmp_path):
    train_cfg, model_cfg, data_cfg = load_train_test_config()
    model_cfg['device'] = 'cpu'
    model_cfg['dtype'] = 'float32'
    train_cfg['steps'] = 1
    train_cfg['save_dir'] = str(tmp_path / "test_checkpoints_empty")
    os.makedirs(train_cfg['save_dir'], exist_ok=True)

    # Patch load_task_dataset specifically for this test to return empty lists
    monkeypatch.setattr(data_utils, 'load_task_dataset', lambda *a, **kw: ([], []))

    # train.py samples from train_prompts. If empty, random.sample will raise ValueError.
    with pytest.raises(ValueError, match=r"(Sample larger than population|Found no prompts|empty dataset)"): # Adjusted match
        train.main(train_cfg, model_cfg, data_cfg)


def test_train_with_single_sample_dataset(monkeypatch, tmp_path):
    train_cfg, model_cfg, data_cfg = load_train_test_config()
    model_cfg['device'] = 'cpu'
    model_cfg['dtype'] = 'float32'
    train_cfg['steps'] = 2
    train_cfg['batch_n'] = 1
    train_cfg['eval_every'] = 1
    train_cfg['save_dir'] = str(tmp_path / "test_checkpoints_single")
    os.makedirs(train_cfg['save_dir'], exist_ok=True)

    # Patch load_task_dataset for single sample
    single_train_ds = [{'question': 'SingleTrainQ', 'answer': 'SingleTrainA'}]
    single_val_ds = [{'question': 'SingleValQ', 'answer': 'SingleValA'}]
    monkeypatch.setattr(data_utils, 'load_task_dataset', 
                        lambda *a, **kw: (single_train_ds, single_val_ds))
    
    train.main(train_cfg, model_cfg, data_cfg)
    expected_ckpt_path = os.path.join(train_cfg['save_dir'], f"model_step_{train_cfg['steps']}.pt")
    assert os.path.exists(expected_ckpt_path), "Checkpoint not saved for single sample dataset training."


# ===============================================================================
# PG/KL TRAINING LOOP TEST
# ===============================================================================
def test_train_loop_with_kl_penalty(tmp_path):
    train_cfg, model_cfg, data_cfg = load_train_test_config()
    model_cfg['device'] = 'cpu'
    model_cfg['dtype'] = 'float32' # Dummy model typically float32
    
    train_cfg['steps'] = 2
    train_cfg['batch_n'] = 1
    train_cfg['eval_every'] = 1
    train_cfg['save_dir'] = str(tmp_path / "test_checkpoints_kl")
    os.makedirs(train_cfg['save_dir'], exist_ok=True)

    # Enable KL penalty
    train_cfg['kl_beta'] = 0.1
    # ref_model_name will use model_cfg['model_path'] by default if null, 
    # which FakeAutoModel will handle by returning another DummyModel instance.
    train_cfg['ref_model_name'] = model_cfg['model_path'] # Explicitly use same path for ref
                                                          # (will be a separate DummyModel instance)

    train.main(train_cfg, model_cfg, data_cfg)
    expected_ckpt_path = os.path.join(train_cfg['save_dir'], f"model_step_{train_cfg['steps']}.pt")
    assert os.path.exists(expected_ckpt_path), "Checkpoint not saved for KL penalty training."


# ===============================================================================
# CONFIG LOADING TEST (Basic check of train_cfg from file)
# ===============================================================================
def test_train_config_values_loaded_correctly():
    train_cfg, model_cfg, data_cfg = load_train_test_config() # Uses helper

    # Check a few key training config values from the default config.yaml
    assert isinstance(train_cfg['batch_n'], int)
    assert train_cfg['batch_n'] == 8 # Default from your config
    assert train_cfg['steps'] == 1000
    assert train_cfg['lr'] == 2.0e-5
    assert train_cfg['save_dir'] == "checkpoints" # Default save_dir
    assert train_cfg['kl_beta'] == 0.0 # Default KL beta
    assert train_cfg['ref_model_name'] is None # Default ref model name


if __name__ == "__main__":
    pytest.main([__file__])