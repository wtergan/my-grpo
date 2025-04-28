import sys, tempfile, types, os
from pathlib import Path

import pytest
import torch
from datasets import Dataset

import train                        # your Phase-3 script
import data_utils as du

# ===============================================================================
# DUMMY CLASSES AND DATA HELPERS
# ===============================================================================
class DummyTB:
    def __init__(self, *_, **__): pass
    add_scalar = lambda *_, **__: None
    close      = lambda self: None

def tiny_dataset():
    return Dataset.from_list([
        {"question": "What is 1+1?", "answer": "2"},
        {"question": "What is 2+2?", "answer": "4"},
    ])

def stub_load_task_dataset(task, *_, **__):
    ds = tiny_dataset()
    return ds, ds                      # train_ds, val_ds

# ===============================================================================
# SMOKE TEST AND BASIC FUNCTIONALITY
# ===============================================================================
@pytest.mark.timeout(60)
def test_train_loop_cpu(monkeypatch):
    # monkey-patch: tiny dataset & no-op TensorBoard
    monkeypatch.setattr(du, "load_task_dataset", stub_load_task_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)

    # craft CLI args for a micro run
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = [
            "train.py",
            "--model_name", "hf-internal-testing/tiny-random-gpt2",
            "--device", "cpu",
            "--steps", "2",
            "--batch_n", "2",
            "--gens_m", "1",
            "--eval_every", "2",
            "--save_dir", tmpdir,
            "--fp16"  # harmless on CPU; ignored
        ]
        monkeypatch.setattr(sys, "argv", cli)

        # run; should not raise
        train.main()

        # checkpoint must exist
        ckpts = list(Path(tmpdir).glob("model_step_2.pt"))
        assert ckpts, "checkpoint file not found"
        ckpt_path = ckpts[0]
        assert ckpt_path.stat().st_size > 0, "checkpoint file is empty"

        # Model reload check
        try:
            state = torch.load(ckpt_path, map_location="cpu")
            assert isinstance(state, dict), "Checkpoint did not load as dict"
        except Exception as e:
            pytest.fail(f"Reloading checkpoint failed: {e}")

        # (Optional) Check that loss is finite (if loss is returned/saved)
        # Not implemented unless train.py is modified to expose loss value

# ===============================================================================
# ARGUMENT PARSER SANITY CHECKS
# ===============================================================================
def test_argparse_defaults():
    parser = train.build_argparser()
    args = parser.parse_args([])
    assert args.model_name
    assert args.task
    assert args.batch_n > 0
    assert args.steps > 0
    assert args.save_dir
    assert args.device in ("cpu", "cuda")

# ===============================================================================
# EDGE CASES: EMPTY AND SINGLE SAMPLE DATASETS
# ===============================================================================
def stub_empty_dataset(*args, **kwargs):
    from datasets import Dataset
    ds = Dataset.from_list([])
    return ds, ds

def test_train_with_empty_dataset(monkeypatch):
    monkeypatch.setattr(du, "load_task_dataset", stub_empty_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = [
            "train.py",
            "--model_name", "hf-internal-testing/tiny-random-gpt2",
            "--device", "cpu",
            "--steps", "2",
            "--batch_n", "2",
            "--gens_m", "1",
            "--eval_every", "2",
            "--save_dir", tmpdir,
        ]
        monkeypatch.setattr(sys, "argv", cli)
        try:
            train.main()
        except Exception as e:
            # Should raise a clear error, not crash with obscure error
            assert "empty" in str(e).lower() or "no data" in str(e).lower() or isinstance(e, (ValueError, RuntimeError)), f"Unexpected error: {e}"
        else:
            pytest.fail("Expected failure on empty dataset, but training completed.")

def stub_single_sample_dataset(*args, **kwargs):
    from datasets import Dataset
    ds = Dataset.from_list([{"question": "What is 1+1?", "answer": "2"}])
    return ds, ds

def test_train_with_single_sample(monkeypatch):
    monkeypatch.setattr(du, "load_task_dataset", stub_single_sample_dataset)
    monkeypatch.setattr(train, "SummaryWriter", DummyTB)
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = [
            "train.py",
            "--model_name", "hf-internal-testing/tiny-random-gpt2",
            "--device", "cpu",
            "--steps", "2",
            "--batch_n", "1",
            "--gens_m", "1",
            "--eval_every", "2",
            "--save_dir", tmpdir,
        ]
        monkeypatch.setattr(sys, "argv", cli)
        # Should not raise
        train.main()
        ckpts = list(Path(tmpdir).glob("model_step_2.pt"))
        assert ckpts, "checkpoint file not found for single sample run"
