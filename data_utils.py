"""
Facilitates the loading of pertinent datasets, formatting prompts, computation of reward (binary).
"""

from __future__ import annotations
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datasets import load_dataset, Dataset
import torch
import json
import re

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskDefinition:
    """
    Task definition: A class template that holds all the information needed to handle a 
    dataset/task. Modular, so can add more fields as needed for that specific task.
    """
    name: str
    load_function: Callable[..., Dataset]
    prompt_function: Callable[[Dict], str]
    answer_extraction: Callable[[str], str]
    answer_key: str
    metric: str = "accuracy"
    extra: dict = field(default_factory=dict)

# ============================================================================
# DATASET LOADING FUNCTIONS
# ============================================================================

def load_gsm8k(split: str = "train") -> Dataset:
    """Loads the cleaned GSM8K split from HuggingFace."""
    return load_dataset("gsm8k", "main", split=split)

def load_countdown_local(split: str = "train") -> Dataset:
    """
    Placeholder for loading the countdown dataset from local JSONL file. Can modify later.
    Looks for data/countdown_<split>.jsonl with keys {prompt, answer}.
    """
    path = Path(__file__).parent / "data" / f"countdown_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Countdown dataset not found at {path}.")
    records = [json.loads(line) for line in path.open()]
    return Dataset.from_list(records)

# ============================================================================
# PROMPT FORMATTING & ANSWER EXTRACTION FUNCTIONS
# ============================================================================

def gsm8k_prompt(example: Dict) -> str:
    return f"""You are a brilliant mathematician. Solve the problem.

Q: {example['question']}

Answer the question. Provide only the numeric result."""

def gsm8k_extraction(text: str) -> str:
    """
    gsm8k answers are in a reasoning - integer answer format, where the integer answer
    is preceded by '####'. Thus, need to extract integer answer from the text for usage.
    """
    answer = re.findall(r'\d+', text)
    return answer[-1] if answer else ""

# ============================================================================
# TASK REGISTRY
# ============================================================================

TASKS: Dict[str, TaskDefinition] = {
    "gsm8k": TaskDefinition(
        name="gsm8k",
        load_function=load_gsm8k,
        prompt_function=gsm8k_prompt,
        answer_extraction=gsm8k_extraction,
        answer_key="answer",
        metric="accuracy",
    ),
    "countdown": TaskDefinition(
        name="countdown",
        load_function=lambda split="train": load_countdown_local(split),
        prompt_function=lambda ex: ex["prompt"],
        answer_extraction=lambda txt: txt.strip(),
        answer_key="answer",
        metric="accuracy",
    ),
}

# ============================================================================
# TASK SPLITTING AND LOADER UTILITIES
# ============================================================================

def get_validation_split_name(task_def: TaskDefinition) -> str | None:
    """Returns the name of the validation split ('validation' or 'test'), or None if not found."""
    for split_name in ["validation", "test"]:
        try:
            task_def.load_function(split_name)
            return split_name
        except (KeyError, FileNotFoundError, ValueError):
            continue
    return None

def load_task_dataset(task_name: str,
                      split: str = "train",
                      seed: int = 42,
                      val_fraction: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Returns (train_ds, val_ds) for the given task.
    If the underlying dataset has 'train'/'validation' or 'train'/'test', we use them.
    Otherwise we do a random split.
    """
    task_def = TASKS[task_name]
    val_split_name = get_validation_split_name(task_def)
    if val_split_name is not None:
        train_ds = task_def.load_function("train")
        val_ds   = task_def.load_function(val_split_name)
    else:
        # Load the full dataset provided split, shuffles and splits into train and val.
        # Val size default will be 0.1 of the full dataset.
        full_ds = task_def.load_function(split)
        full_ds = full_ds.shuffle(seed=seed)
        val_size = int(len(full_ds) * val_fraction)
        train_ds = full_ds.select(range(len(full_ds) - val_size))
        val_ds   = full_ds.select(range(len(full_ds) - val_size, len(full_ds)))
    return train_ds, val_ds

# ============================================================================
# PROMPT BUILDING, TARGET EXTRACTION, REWARD COMPUTATION
# ============================================================================

def build_prompts(dataset: Dataset,
                  task_name: str) -> List[str]:
    task_def = TASKS[task_name]
    return [task_def.prompt_function(example) for example in dataset]

def target_extraction(dataset: Dataset,
                      task_name: str) -> List[str]:
    task_def = TASKS[task_name]
    return [example[task_def.answer_key] for example in dataset]

def compute_binary_reward(predictions: List[str],
                   targets: List[str],
                   task_name: str) -> torch.FloatTensor:
    """
    Computation of a binary accuracy reward for each prediction-target pair.
    Uses task-specific answer extraction logic for fair comparison.
    returns shape (N,) float32 tensor ∈ {0,1}.
    """
    task_def = TASKS[task_name]
    assert len(predictions) == len(targets)
    rewards = []
    for pred, tgt in zip(predictions, targets):
        pred_norm = task_def.answer_extraction(pred)
        tgt_norm  = task_def.answer_extraction(tgt)
        rewards.append(float(pred_norm == tgt_norm))
    return torch.tensor(rewards, dtype=torch.float32)
