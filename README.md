My own implementation of **Group Relative Policy Optimization** for language model post-training.

## Why?
To grok RL-finetune internals for educational purposes.s

## Project structure:
grpo_core.py      # GRPO math: generation, reward-norm, loss.
data_utils.py     # dataset fetch, prompt format, reward fn.
train.py          # orchestration loop.
test.py           # interactive or batch evaluate.