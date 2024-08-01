<h1 align="center">Flab-Pruner: Towards Greener Yet Powerful Code Intelligence via Structural Pruning</h1>

<p align="left">
    ⭐️&nbsp;<a href="#about">About</a>
    | 🚀&nbsp;<a href="#quick-start">Quick start</a>
    | 📊&nbsp;<a href="#evaluation">Evaluation</a>
    | ⚠️&nbsp;<a href="#bias-risks-and-limitations">Limitations</a>
</p>

## About

We introduce Flab-Pruner, a structural pruning approach that enhances efficiency and sustainability without compromising the performance of Code LLMs.

- **CodeQwen1.5-Pruned Models:** [CodeQwen1.5-Pruned](https://huggingface.co/Flab-Pruner/Flab-CQ-5.7B-instruct)
- **Nxcode-CQ-Pruned Models:** [Nxcode-CQ-pruned](https://huggingface.co/Flab-Pruner/Flab-Nxcode-5.7B-instruct)
- **Dataset:** [Flab-Pruner/CodeHarmony](https://huggingface.co/datasets/Flab-Pruner/CodeHarmony)


## Quick start

### Vocab Pruning
>see examples in vocab_prune.py

### Layer Pruning
>see examples in greedy_prune.py

### FFN Pruning
>see examples in ffn_prune.py

### Post Training
>see examples in run_lora.py

### How to use tuned model
Here is an example to get started with CodeQwen1.5-5.58B-Instruct-V1 using the [transformers](https://huggingface.co/docs/transformers/index) library:

```python
import transformers
import torch

```

## Evaluation

## Bias, Risks, and Limitations

## Cite as

```
@misc{Flab-Pruner,
  author       = {Flab-Pruner},
  title = {Flab-Pruner: Towards Greener Yet Powerful Code Intelligence via Structural Pruning},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Flab-Pruner/Flab-Pruner}},
  year = 2024,
}
