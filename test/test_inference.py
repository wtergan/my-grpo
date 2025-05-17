import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import yaml
import pytest
import json # For checking output file content

# Import the module to test, aliased as T for brevity from original
import inference as T 
# Dummies will come from conftest.py

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.yaml')

def load_test_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    # Deepcopy or carefully modify for tests to avoid state leakage if needed
    # For these tests, direct modification of a loaded dict slice is fine.
    return config['testing'], config['model'], config['data']

@pytest.fixture(autouse=True)
def patch_inference_hf_dependencies(monkeypatch, request):
    # Patch AutoTokenizer and AutoModelForCausalLM where inference.py imports them
    FakeAutoTokenizer = request.getfixturevalue('FakeAutoTokenizer')
    FakeAutoModel = request.getfixturevalue('FakeAutoModel')
    
    monkeypatch.setattr(T, 'AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr(T, 'AutoModelForCausalLM', FakeAutoModel)

# ===============================================================================
# REPL MODE TESTS
# ===============================================================================
def test_repl_keyboard_interrupt(monkeypatch):
    test_cfg, model_cfg, data_cfg = load_test_config()
    model_cfg['device'] = 'cpu' # Ensure CPU for test environment
    test_cfg['prompts'] = None # Ensure REPL mode

    def fake_input(prompt_text): # prompt_text is the "Prompt: " string
        raise KeyboardInterrupt()
    monkeypatch.setattr('builtins.input', fake_input)
    
    with pytest.raises(SystemExit) as e:
        T.main(test_cfg, model_cfg, data_cfg)
    assert e.value.code == 0


def test_repl_eof_error(monkeypatch):
    test_cfg, model_cfg, data_cfg = load_test_config()
    model_cfg['device'] = 'cpu'
    test_cfg['prompts'] = None 

    def fake_input(prompt_text):
        raise EOFError()
    monkeypatch.setattr('builtins.input', fake_input)

    with pytest.raises(SystemExit) as e:
        T.main(test_cfg, model_cfg, data_cfg)
    assert e.value.code == 0


def test_repl_single_prompt_then_empty_exit(monkeypatch, capsys):
    test_cfg, model_cfg, data_cfg = load_test_config()
    model_cfg['device'] = 'cpu'
    test_cfg['prompts'] = None
    test_cfg['out'] = None # Print to stdout for REPL

    user_inputs = iter(["What is 2+2?", ""]) # First prompt, then empty to exit
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))
    
    # No SystemExit expected if loop finishes due to empty input after first.
    # The script sys.exit(0) if prompt is empty.
    with pytest.raises(SystemExit) as e:
        T.main(test_cfg, model_cfg, data_cfg)
    assert e.value.code == 0
    
    captured = capsys.readouterr()
    # Dummy model generate will produce something based on DummyTokenizer.batch_decode
    # Check if JSON-like output for the first prompt was printed
    assert '"prompt": "What is 2+2?"' in captured.out
    assert '"completion":' in captured.out # Actual completion text depends on DummyModel


# ===============================================================================
# BATCH MODE TESTS
# ===============================================================================
def test_batch_mode_with_file(tmp_path, monkeypatch):
    test_cfg, model_cfg, data_cfg = load_test_config()
    model_cfg['device'] = 'cpu'

    prompts_content = "What is 2+2?\nWhat is the capital of France?\n"
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(prompts_content)
    
    output_file = tmp_path / "output.jsonl" # Original used .jsonl, main script uses .json

    test_cfg['prompts'] = str(prompts_file)
    test_cfg['out'] = str(output_file)
    test_cfg['max_new_tokens'] = 32 # As in original test

    T.main(test_cfg, model_cfg, data_cfg)
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        results = json.load(f)
    assert len(results) == 2
    assert results[0]['prompt'] == "What is 2+2?"
    assert 'completion' in results[0]


def test_batch_mode_missing_prompts_file(tmp_path, monkeypatch, capsys):
    test_cfg, model_cfg, data_cfg = load_test_config()
    model_cfg['device'] = 'cpu'

    prompts_file = tmp_path / "does_not_exist.txt" # File does not exist
    output_file = tmp_path / "output.jsonl"

    test_cfg['prompts'] = str(prompts_file)
    test_cfg['out'] = str(output_file)

    T.main(test_cfg, model_cfg, data_cfg) # Should not crash, should print error
    
    captured = capsys.readouterr()
    assert f"Error: Prompts file not found: {prompts_file}" in captured.out
    assert not output_file.exists() # As per logic in main, if prompts_file not found, it returns.


def test_batch_mode_empty_prompts_file(tmp_path, monkeypatch):
    test_cfg, model_cfg, data_cfg = load_test_config()
    model_cfg['device'] = 'cpu'

    prompts_file = tmp_path / "empty_prompts.txt"
    prompts_file.write_text("") # Empty file
    
    output_file = tmp_path / "output.jsonl"

    test_cfg['prompts'] = str(prompts_file)
    test_cfg['out'] = str(output_file)

    T.main(test_cfg, model_cfg, data_cfg)
    
    # Logic: if prompts list is empty, and out_file is specified, it writes an empty list.
    assert output_file.exists()
    with open(output_file, 'r') as f:
        content = f.read()
        assert json.loads(content) == []


# ===============================================================================
# CONFIG LOADING TEST (Basic check, not a test of T.main)
# ===============================================================================
def test_inference_config_values_from_file():
    test_cfg, model_cfg, data_cfg = load_test_config() # Uses the helper
    
    # Check some specific values from config.yaml's 'testing' section
    assert test_cfg['model_name'] == "Qwen/Qwen2.5-3B-Instruct" # From default config
    assert test_cfg['device'] == "cuda" # Default from config
    assert test_cfg['max_new_tokens'] == 128
    assert test_cfg['task'] == "gsm8k"
    assert test_cfg['ckpt'] is None 
    assert test_cfg['prompts'] is None
    assert test_cfg['out'] is None

if __name__ == "__main__":
    pytest.main([__file__])