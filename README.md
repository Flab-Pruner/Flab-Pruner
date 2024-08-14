<h1 align="center">Flab-Pruner: Towards Greener Yet Powerful Code Intelligence via Structural Pruning</h1>

<p align="left">
      🔥&nbsp;<a href="#news">News</a>
    | ⭐️&nbsp;<a href="#about">About</a>
    | 🚀&nbsp;<a href="#quick-start">Quick start</a>
    | 📊&nbsp;<a href="#evaluation">Evaluation</a>
    | ⚠️&nbsp;<a href="#bias-risks-and-limitations">Limitations</a>
</p>

## News

We design an **new algorithm** for data selection to address the problem of model performance recovery after pruning. 

<img src="./demo.png" alt="./demo.png" style="zoom:5%;" />

We release the new 5.7B-size model Flab-CQ-V1.5(https://huggingface.co/Flab-Pruner/Flab-CQ-V1.5) based on this algorithm, which can get **83.54** pass@1 on HumanEval.

## About

We introduce Flab-Pruner, a structural pruning approach that enhances efficiency and sustainability without compromising the performance of Code LLMs.

- **CodeQwen1.5-Pruned Models Version 1.5:** [Flab-CQ-V1.5](https://huggingface.co/Flab-Pruner/Flab-CQ-V1.5)
- **CodeQwen1.5-Pruned Models:** [Flab-CQ](https://huggingface.co/Flab-Pruner/Flab-CQ-5.7B-instruct)
- **Nxcode-CQ-Pruned Models Version 1.5:** [Flab-Nxcode-V1.5](https://huggingface.co/Flab-Pruner/Flab-Nxcode-V1.5)
- **Nxcode-CQ-Pruned Models:** [Flab-Nxcode](https://huggingface.co/Flab-Pruner/Flab-Nxcode-5.7B-instruct)
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

## Evaluation

### Evaluation on HumanEval, OpenEval, and CodeHarmony

| Model                 | Params | Code | (Pass@1) |      |      | CoT    | (BLEU) |        |      | Output | (EM) |      |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ------ | ---- | ------ | ---- | ---- |
|                       |      | HE   | OE   | CH   | Avg. | HE-CoT | OE-CoT | CH-CoT | Avg. | Crux-O | CH-O | Avg. |
| CodeQwen              | 7.3B | 76.83 | **41.57** | 61.44 | 59.95 | 33.88 | **40.48** | 24.19 | **32.85** | 37.13 | 77.21 | 57.17 |
| Flab-CQ-base          | 5.7B | 70.12 | 41.01 | 62.75 | 57.96 | 29.92 | 35.62 | 20.70 | 28.75 | 32.50 | 73.67 | 53.09 |
| Flab-CQ-V1  | 5.7B | 76.22 | 38.76 | 64.71 | 59.90 | 33.22 | 36.41 | 23.89 | 31.17 | 39.00 | 76.99 | 58.00 |
| Flab-CQ-V1.5 | 5.7B | **83.54** | 39.33 | **67.32** | **63.40** | **33.92** | 32.61 | **25.23** | 30.59 | **43.75** | **77.43** | **60.59** |
| Nxcode | 7.3B | 76.22 | 43.26 | 62.75 | 60.74 | 33.05 | 40.29 | 24.17 | 32.50 | 37.25 | 76.77 | 57.01 |
| Flab-Nxcode-base | 5.7B | 72.56 | 38.20 | 60.13 | 56.96 | 30.20 | 35.14 | 20.71 | 28.68 | 33.38 | 74.12 | 53.75 |
| Flab-Nxcode-V1 | 5.7B | **79.88** | 39.33 | **66.01** | **61.74** | **34.92** | 37.06 | **26.27** | **32.75** | **41.25** | **79.20** | **60.23** |

### Evaluation on RECODE

| Model            | Perturbed |           |           |            |
| ---------------- | --------- | --------- | --------- | ---------- |
|                  | format    | func_name | natgen    | nlaugenter |
| CodeQwen         | 82.93     | 78.66     | 82.93     | 62.80      |
| Flab-CQ-base     | 74.39     | 74.39     | 77.44     | 57.32      |
| Flab-CQ-V1       | 81.71     | 74.39     | 76.22     | 64.34      |
| Flab-CQ-V1.5     | **85.37** | **79.88** | **83.54** | **68.90**  |
| Nxcode           | **82.32** | 76.22     | **84.15** | 65.24      |
| Flab-Nxcode-base | 75.61     | 72.56     | 76.83     | 57.93      |
| Flab-Nxcode-V1   | 79.88     | **76.83** | 74.39     | **67.07**  |



### Evaluation on EvoEval
| Model            | Tool_Use | Combine | Subtle | Creative | Difficult |
| ---------------- | -------- | ------- | ------ | -------- | --------- |
| CodeQwen         | 48       | 23      | 62     | **36**   | **36**    |
| Flab-CQ-base     | 48       | 15      | 57     | 21       | 26        |
| Flab-CQ-V1       | 57       | 9       | **64** | 29       | 23        |
| Flab-CQ-V1.5     | **59**   | **24**  | 61     | 35       | 30        |
| Nxcode           | 49       | **24**  | 62     | 33       | **35**    |
| Flab-Nxcode-base | 49       | 16      | 57     | 24       | 27        |
| Flab-Nxcode-V1   | **56**   | 12      | **63** | **34**   | 24        |


## Bias, Risks, and Limitations

The CodeHarmony dataset was constructed by mining the existing instruction fine-tuning datasets, and despite our best efforts to ensure the correctness of the code by constructing test cases for it, it is still possible to have incorrect code. Therefore, users should still be careful when using these datasets.

## Cite as

```
@misc{Flab-Pruner,
  author = {Flab-Pruner},
  title = {Flab-Pruner: Towards Greener Yet Powerful Code Intelligence via Structural Pruning},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Flab-Pruner/Flab-Pruner}},
  year = 2024,
}