# ===============================================================================
# IMPORTS AND SETUP
# ===============================================================================
import os
import yaml
import inference as T
import pytest

# Helper to load config for tests
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ===============================================================================
# REPL MODE TESTS
# ===============================================================================
# Test 1: REPL starts and exits gracefully on Enter (already provided by user)
# Test 2: REPL exits gracefully on Ctrl+C (KeyboardInterrupt)
def test_repl_keyboard_interrupt(monkeypatch):
    config = load_config()
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    def fake_input(prompt):
        raise KeyboardInterrupt()
    monkeypatch.setattr('builtins.input', fake_input)
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    with pytest.raises(SystemExit) as e:
        T.main(test_cfg, model_cfg, data_cfg)
    assert e.value.code == 0

# Test 3: REPL exits gracefully on Ctrl+D (EOFError)
def test_repl_eof_error(monkeypatch):
    config = load_config()
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    def fake_input(prompt):
        raise EOFError()
    monkeypatch.setattr('builtins.input', fake_input)
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    with pytest.raises(SystemExit) as e:
        T.main(test_cfg, model_cfg, data_cfg)
    assert e.value.code == 0

# Test 4: REPL generates a completion for a single prompt and exits
def test_repl_single_prompt(monkeypatch):
    config = load_config()
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    prompts = iter(["What is 2+2?", ""])
    monkeypatch.setattr('builtins.input', lambda _: next(prompts))
    generated_outputs = []
    def mock_generate(*args, **kwargs):
        generated_outputs.append("mocked_output_for_first_prompt")
        return "mocked_output_for_first_prompt"

    monkeypatch.setattr(T, 'generate', mock_generate) 
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    with pytest.raises(SystemExit) as e: 
        T.main(test_cfg, model_cfg, data_cfg)
    assert e.value.code == 0
    assert len(generated_outputs) == 1

# ===============================================================================
# BATCH MODE TESTS
# ===============================================================================
# Test 5: Batch mode with file (minimal smoke test)
def test_batch_mode(tmp_path, monkeypatch):
    config = load_config()
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("What is 2+2?\nWhat is the capital of France?\n")
    output_file = tmp_path / "output.jsonl"
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "dummy")
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    test_cfg['prompts'] = str(prompts_file)
    test_cfg['out'] = str(output_file)
    test_cfg['max_new_tokens'] = 32
    T.main(test_cfg, model_cfg, data_cfg)
    assert output_file.exists()

# Test: Batch mode with missing prompts file (should not raise an exception)
def test_batch_mode_missing_file(tmp_path, monkeypatch):
    config = load_config()
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    prompts_file = tmp_path / "does_not_exist.txt"
    output_file = tmp_path / "output.jsonl"
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "dummy")
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    test_cfg['prompts'] = str(prompts_file)
    test_cfg['out'] = str(output_file)
    test_cfg['max_new_tokens'] = 32
    T.main(test_cfg, model_cfg, data_cfg)
    assert not output_file.exists()

# Test: Batch mode with empty prompts file (should not crash)
def test_batch_mode_empty_file(tmp_path, monkeypatch):
    config = load_config()
    test_cfg = config['testing']
    model_cfg = config['model']
    data_cfg = config['data']
    prompts_file = tmp_path / "empty.txt"
    prompts_file.write_text("")
    output_file = tmp_path / "output.jsonl"
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "dummy")
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    test_cfg['prompts'] = str(prompts_file)
    test_cfg['out'] = str(output_file)
    test_cfg['max_new_tokens'] = 32
    T.main(test_cfg, model_cfg, data_cfg)
    assert output_file.exists()

# ===============================================================================
# CONFIG LOADING TEST
# ===============================================================================
# Test: Load config.yaml and assert expected keys and values for the testing section
def test_inference_config_loading():
    config = load_config()
    # Check testing section
    assert 'testing' in config
    test_cfg = config['testing']
    assert test_cfg['model_name'] == "Qwen/Qwen2.5-3B-Instruct"
    assert test_cfg['device'] == "cuda"
    assert test_cfg['max_new_tokens'] == 128
    assert test_cfg['task'] == "gsm8k"
    # Spot check nullables
    assert test_cfg['ckpt'] is None
    assert test_cfg['prompts'] is None
    assert test_cfg['out'] is None
