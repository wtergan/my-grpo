# ===============================================================================
# IMPORTS AND SETUP
# ===============================================================================
import subprocess, sys, textwrap
import pytest
import yaml
import os

# ===============================================================================
# REPL MODE TESTS
# ===============================================================================
# Test 1: REPL starts and exits gracefully on Enter (already provided by user)
# Test 2: REPL exits gracefully on Ctrl+C (KeyboardInterrupt)
def test_repl_keyboard_interrupt():
    code = textwrap.dedent('''
        import inference as T
        import sys, builtins
        def fake_input(prompt):
            raise KeyboardInterrupt()
        builtins.input = fake_input
        T.main()
    ''')
    subprocess.run([sys.executable, "-c", code], check=True)

# Test 3: REPL exits gracefully on Ctrl+D (EOFError)
def test_repl_eof_error():
    code = textwrap.dedent('''
        import inference as T
        import sys, builtins
        def fake_input(prompt):
            raise EOFError()
        builtins.input = fake_input
        T.main()
    ''')
    subprocess.run([sys.executable, "-c", code], check=True)

# Test 4: REPL generates a completion for a single prompt and exits
def test_repl_single_prompt(monkeypatch):
    import inference as T
    prompts = iter(["What is 2+2?", ""])
    monkeypatch.setattr('builtins.input', lambda _: next(prompts))
    # Patch generate to avoid running a real model
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "4")
    # Patch model/tokenizer to dummy objects
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    monkeypatch.setattr(
        T, 'build_parser',
        lambda: type('P', (), {
            'parse_args': lambda *a, **kw: type('A', (), {
                'model_name': '', 'ckpt': None, 'device': 'cpu', 'prompts': None, 'task': '', 'out': None, 'max_new_tokens': 10, 'targets': None
            })()
        })()
    )
    T.main()

# ===============================================================================
# BATCH MODE TESTS
# ===============================================================================
# Test 5: Batch mode with file (minimal smoke test)
def test_batch_mode(tmp_path, monkeypatch):
    import inference as T
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("What is 2+2?\nWhat is the capital of France?\n")
    # Patch generate, load_model, build_parser as above
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "dummy")
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    monkeypatch.setattr(
        T, 'build_parser',
        lambda: type('P', (), {
            'parse_args': lambda *a, **kw: type('A', (), {
                'model_name': '', 'ckpt': None, 'device': 'cpu', 'prompts': str(prompts_file), 'task': '', 'out': None, 'max_new_tokens': 10, 'targets': None
            })()
        })()
    )
    T.main()

# ===============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# ===============================================================================
# Test: Batch mode with missing prompts file (should raise FileNotFoundError or similar)
def test_batch_mode_missing_file(tmp_path, monkeypatch):
    import inference as T
    prompts_file = tmp_path / "does_not_exist.txt"
    # Patch generate, load_model, build_parser as above
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "dummy")
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    monkeypatch.setattr(
        T, 'build_parser',
        lambda: type('P', (), {
            'parse_args': lambda *a, **kw: type('A', (), {
                'model_name': '', 'ckpt': None, 'device': 'cpu', 'prompts': str(prompts_file), 'task': '', 'out': None, 'max_new_tokens': 10, 'targets': None
            })()
        })()
    )
    with pytest.raises((FileNotFoundError, OSError, IOError)):
        T.main()

# Test: Batch mode with empty prompts file (should not crash)
def test_batch_mode_empty_file(tmp_path, monkeypatch):
    import inference as T
    prompts_file = tmp_path / "empty.txt"
    prompts_file.write_text("")
    monkeypatch.setattr(T, 'generate', lambda *a, **kw: "dummy")
    monkeypatch.setattr(T, 'load_model', lambda *a, **kw: (None, None))
    monkeypatch.setattr(
        T, 'build_parser',
        lambda: type('P', (), {
            'parse_args': lambda *a, **kw: type('A', (), {
                'model_name': '', 'ckpt': None, 'device': 'cpu', 'prompts': str(prompts_file), 'task': '', 'out': None, 'max_new_tokens': 10, 'targets': None
            })()
        })()
    )
    # Should not raise
    T.main()

# ===============================================================================
# CONFIG LOADING TEST
# ===============================================================================
# Test: Load config.yaml and assert expected keys and values for the testing section
def test_inference_config_loading():
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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
