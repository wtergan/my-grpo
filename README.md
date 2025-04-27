Toy-level re-implementation of **Group Relative Policy Optimization** for language model post-training.

## Why?
To grok RL-finetune internals without the scaffolding (Ray, RLHF pipelines, etc.).

## Project structure:
grpo_core.py      # GRPO math: generation, reward-norm, loss.
data_utils.py     # dataset fetch, prompt format, reward fn.
train.py          # orchestration loop.
test.py           # interactive or batch evaluate.