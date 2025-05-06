# ===============================================================================
# IMPORTS AND SETUP
# ===============================================================================
import sys
import os
import tempfile
import yaml
import pytest
import train
import data_utils as du

# Dummy classes/functions for monkeypatching
class DummyTB:
    def __init__(self, *a, **kw): pass
    def add_scalar(self, *a, **kw): pass
    def close(self): pass

def stub_load_task_dataset(task):
    # Minimal stub dataset
    d = [{"input": "x", "output": "y"}]
    return d, d

def stub_empty_dataset(task):
    return [], []

def stub_single_sample_dataset(task):
    d = [{"input": "x", "output": "y"}]
    return d, d

# Helper to load config and train_cfg
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.yaml')
def load_configs():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config, config['training']

# ===============================================================================
# SMOKE TESTS AND BASIC FUNCTIONALITY
# ===============================================================================
@pytest.mark.timeout(60)
def test_train_loop_cpu(monkeypatch):
    monkeypatch.setattr(du, "load_task_dataset", stub_load_task_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)
    config, train_cfg = load_configs()
    # Minimal run: just check it doesn't crash
    train.main(config, train_cfg)

# ===============================================================================
# EDGE CASES: EMPTY AND SINGLE SAMPLE DATASETS
# ===============================================================================
# Test training with empty dataset:
def test_train_with_empty_dataset(monkeypatch):
    monkeypatch.setattr(du, "load_task_dataset", stub_empty_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)
    config, train_cfg = load_configs()
    try:
        train.main(config, train_cfg)
    except Exception as e:
        assert "empty" in str(e).lower() or "no data" in str(e).lower() or isinstance(e, (ValueError, RuntimeError)), f"Unexpected error: {e}"

# Test training with single sample dataset:
def test_train_with_single_sample(monkeypatch):
    monkeypatch.setattr(du, "load_task_dataset", stub_single_sample_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)
    config, train_cfg = load_configs()
    train.main(config, train_cfg)

# ===============================================================================
# PG/KL TRAINING LOOP TEST
# ===============================================================================
# Test training with PG vs KL:
def test_train_loop_pg_vs_kl(monkeypatch):
    monkeypatch.setattr(du, "load_task_dataset", stub_load_task_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)
    config, train_cfg = load_configs()
    # PG mode
    train_cfg['kl_beta'] = 0.0
    train.main(config, train_cfg)
    # KL mode
    train_cfg['kl_beta'] = 0.5
    train.main(config, train_cfg)

# ===============================================================================
# CONFIG LOADING TEST
# ===============================================================================
def test_train_config_loading():
    config, train_cfg = load_configs()
    assert 'model' in config
    assert config['model']['model_path'] == "Qwen/Qwen2.5-3B-Instruct"
    assert config['model']['dtype'] == "bfloat16"
    assert 'data' in config
    assert 'data_path' in config['data']
    assert 'training' in config
    assert type(train_cfg['batch_n']) is int
    assert train_cfg['save_dir'] == "checkpoints"
    assert train_cfg['kl_beta'] == 0.0
    assert train_cfg['ref_model_name'] is None
