import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from pruning_layer import ShortTransformer
from pruning_layer.utils import (
    draw_diagram,
    get_scored_blocks,
    get_best_pruning_start,
    get_topk_pruning_start
)
from transformers import AutoTokenizer
import pandas as pd
from pruning_layer.dist import *


def greedy_prune(model_name = "Nxcode-CQ-7B-orpo", n_layer=1):

    df = pd.read_json('./dataset/code_generation/CodeHarmony_valid.jsonl', lines=True)
    prompt_list = df['prompt'].tolist()
    canonical_solution_list = df['canonical_solution'].tolist()
    data_list = []
    for prompt, cs in zip(prompt_list, canonical_solution_list):
        data = prompt.strip()
        data = data + '\n'
        data_list.append(data)

    model, layer_map = ShortTransformer.from_pretrained(model_name, 
                                            torch_dtype=torch.float16, 
                                            device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.set_metric(bi_score_last)
    results = model.analyse_layers(
        data_list=data_list,
        tokenizer=tokenizer,
        use_chat_template=False,
        max_length=512,
    )
    start_layer = get_topk_pruning_start(results, topk=1)[0][0]
    print(start_layer)
    short_model = model.prune(layer_map, start_layer, 1)
    for i in range(n_layer-1):
        del model
        model, layer_map = ShortTransformer.from_model(short_model)
        model.set_metric(bi_score_last)
        results = model.analyse_layers(
            data_list=data_list,
            tokenizer=tokenizer,
            use_chat_template=False,
            max_length=512,
        )
        start_layer = get_topk_pruning_start(results, topk=1)[0][0]
        print(start_layer)
        short_model = model.prune(layer_map, start_layer, 1)
    del model
    return short_model, tokenizer

if __name__ == '__main__':
    model, tokenizer = greedy_prune(model_name="./save_models/AutoCoder_QW_7B_vocab", n_layer=4)
    tokenizer.save_pretrained("./save_models/AutoCoder_QW_7B_vocab_layer4")
    model.save_pretrained("./save_models/AutoCoder_QW_7B_vocab_layer4")