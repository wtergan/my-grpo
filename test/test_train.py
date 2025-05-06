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

# === BEGIN: Dummy classes and autouse fixture for monkeypatching ===
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

@pytest.fixture(autouse=True)
def patch_train_dependencies(monkeypatch):
    monkeypatch.setattr('train.AutoTokenizer', type('AutoTokenizer', (), {'from_pretrained': staticmethod(lambda *a, **kw: DummyTokenizer())}))
    monkeypatch.setattr('train.AutoModelForCausalLM', type('AutoModelForCausalLM', (), {'from_pretrained': staticmethod(lambda *a, **kw: DummyModel())}))
    monkeypatch.setattr('train.du', 'load_task_dataset', lambda *a, **kw: ([{'input': 'x', 'output': 'y'}], [{'input': 'x', 'output': 'y'}]))
# === END: Dummy classes and autouse fixture ===

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

# Helper to load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.yaml')
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ===============================================================================
# SMOKE TESTS AND BASIC FUNCTIONALITY
# ===============================================================================
@pytest.mark.timeout(60)
def test_train_loop_cpu(monkeypatch):
    config = load_config()
    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg['save_dir'] = 'test_checkpoints'  # Ensure writable
    train_cfg['seed'] = 123
    train_cfg['batch_n'] = 2
    train_cfg['gens_m'] = 2
    train_cfg['steps'] = 3
    train_cfg['eval_every'] = 1
    train_cfg['lr'] = 1e-4
    model_cfg['device'] = 'cpu'
    train.main(train_cfg, model_cfg, data_cfg)
    assert os.path.exists('test_checkpoints/model_step_3.pt')

# ===============================================================================
# EDGE CASES: EMPTY AND SINGLE SAMPLE DATASETS
# ===============================================================================
# Test training with empty dataset:
def test_train_with_empty_dataset(monkeypatch):
    config = load_config()
    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg['save_dir'] = 'test_checkpoints_empty'
    train_cfg['seed'] = 123
    train_cfg['batch_n'] = 1
    train_cfg['gens_m'] = 1
    train_cfg['steps'] = 1
    train_cfg['eval_every'] = 1
    train_cfg['lr'] = 1e-4
    model_cfg['device'] = 'cpu'
    # Patch dataset loader to return empty
    monkeypatch.setattr(train.du, 'load_task_dataset', lambda *a, **kw: ([], []))
    try:
        train.main(train_cfg, model_cfg, data_cfg)
    except ValueError as e:
        assert 'empty' in str(e).lower()
    except Exception as e:
        assert False, f"Unexpected error: {e}"

# Test training with single sample dataset:
def test_train_with_single_sample(monkeypatch):
    config = load_config()
    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg['save_dir'] = 'test_checkpoints_single'
    train_cfg['seed'] = 123
    train_cfg['batch_n'] = 1
    train_cfg['gens_m'] = 1
    train_cfg['steps'] = 1
    train_cfg['eval_every'] = 1
    train_cfg['lr'] = 1e-4
    model_cfg['device'] = 'cpu'
    dummy_ds = [{'question': 'What is 1+1?', 'answer': '2'}]
    monkeypatch.setattr(train.du, 'load_task_dataset', lambda *a, **kw: (dummy_ds, dummy_ds))
    train.main(train_cfg, model_cfg, data_cfg)
    assert os.path.exists('test_checkpoints_single/model_step_1.pt')

# ===============================================================================
# PG/KL TRAINING LOOP TEST
# ===============================================================================
# Test training with PG vs KL:
def test_train_loop_pg_vs_kl(monkeypatch):
    config = load_config()
    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg['save_dir'] = 'test_checkpoints_pgkl'
    train_cfg['seed'] = 123
    train_cfg['batch_n'] = 2
    train_cfg['gens_m'] = 2
    train_cfg['steps'] = 2
    train_cfg['eval_every'] = 1
    train_cfg['lr'] = 1e-4
    train_cfg['kl_beta'] = 0.5
    train_cfg['ref_model_name'] = model_cfg['model_path']
    model_cfg['device'] = 'cpu'
    train.main(train_cfg, model_cfg, data_cfg)
    assert os.path.exists('test_checkpoints_pgkl/model_step_2.pt')

# ===============================================================================
# CONFIG LOADING TEST
# ===============================================================================
def test_train_config_loading():
    config = load_config()
    assert 'model' in config
    assert config['model']['model_path'] == "Qwen/Qwen2.5-3B-Instruct"
    assert config['model']['dtype'] == "bfloat16"
    assert 'data' in config
    assert 'data_path' in config['data']
    assert 'training' in config
    train_cfg = config['training']
    assert type(train_cfg['batch_n']) is int
    assert train_cfg['save_dir'] == "checkpoints"
    assert train_cfg['kl_beta'] == 0.0
    assert train_cfg['ref_model_name'] is None
