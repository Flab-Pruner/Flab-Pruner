# @TODO check torch, transformers installs
# @TODO check model compability
import numpy as np
import torch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # 减去最大值提高数值稳定性
    return e_x / e_x.sum(axis=0)

def get_best_pruning_start(result, block_size: int) -> int:
    layer_count = result.shape[0]
    assert (
        block_size < layer_count and block_size > 0
    ), f"Expected `block_size` value between 1 and {layer_count -1}, got {block_size}."
    layer_result = result[block_size, : layer_count - block_size]

    start_layer = np.argmin(layer_result)
    return start_layer


def get_topk_pruning_start(result, topk: int) -> list:
    # print(np.array(result[0]))
    candidate_scores = {}
    importances = softmax(np.array(result[0]))
    # importances = np.array(result[0])
    candidate_layers = np.argsort(importances).tolist()[:topk]
    for candi in candidate_layers:
        candidate_scores[str(candi)] = importances[candi]
    return candidate_layers, candidate_scores

def get_scored_blocks(result, return_md=True, threshold=float("inf")) -> dict:
    layer_count = result.shape[0]
    stats = {}
    for i in range(1, layer_count):
        layer_result = result[i, : layer_count - i]
        start_layer = np.argmin(layer_result)
        score = layer_result[start_layer]
        if score <= threshold:
            stats[i] = {"start_layer": start_layer, "score": score}

    if not return_md:
        return stats

    stats_md = "| Block_size | Removed_layers | Score (avg dist)|\n"
    stats_md += "| -------- | ------- | -------- |\n"
    for k, v in stats.items():
        stats_md += f"| {k} | {v['start_layer']}-{v['start_layer']+k-1} | {round(v['score'], 3)}|\n"

    return stats_md


def add_bias(model, start_layer):
    print(model.config.hidden_size)
    new_gate_proj = torch.nn.Linear(in_features=model.config.hidden_size, 
                                out_features=5504, 
                                bias=True, 
                                dtype=model.dtype)
    new_gate_proj.weight.data = model.model.layers[start_layer-1].mlp.gate_proj.weight.data
    model.model.layers[start_layer-1].mlp.gate_proj = new_gate_proj

    new_up_proj = torch.nn.Linear(in_features=model.config.hidden_size, 
                                    out_features=5504, 
                                    bias=True, 
                                    dtype=model.dtype)
    new_up_proj.weight.data = model.model.layers[start_layer-1].mlp.up_proj.weight.data
    model.model.layers[start_layer-1].mlp.up_proj = new_up_proj

    new_down_proj = torch.nn.Linear(in_features=5504, 
                                    out_features=model.config.hidden_size, 
                                    bias=True, 
                                    dtype=model.dtype)
    new_down_proj.weight.data = model.model.layers[start_layer-1].mlp.down_proj.weight.data
    model.model.layers[start_layer-1].mlp.down_proj = new_down_proj
    return model