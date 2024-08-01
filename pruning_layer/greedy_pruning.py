from pruning_layer import ShortTransformer
from pruning_layer.utils import (
    get_topk_pruning_start
)
from transformers import AutoTokenizer
import pandas as pd
from pruning_layer.dist import *
from pre_train.LLAMA_model import LLAMA

def greedy_prune(model_name = "Nxcode-CQ-7B-orpo", n_layer=1, optim=''):

    df = pd.read_json('./dataset/code_generation/CodeHarmony_valid.jsonl', lines=True)
    prompt_list = df['prompt'].tolist()
    canonical_solution_list = df['canonical_solution'].tolist()
    data_list = []
    for prompt, cs in zip(prompt_list, canonical_solution_list):
        data = prompt
        data = data.strip()
        data_list.append(data)

    # model, layer_map = ShortTransformer.from_pretrained(model_name, 
    #                                         torch_dtype=torch.float16, 
    #                                         device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model.set_metric(bi_score_last)
    # results = model.analyse_layers(
    #     data_list=data_list,
    #     tokenizer=tokenizer,
    #     use_chat_template=False,
    #     max_length=512,
    # )
    # start_layer = get_topk_pruning_start(results, topk=1)[0][0]
    # print(start_layer)
    # short_model = model.prune(layer_map, start_layer, 1)
    # del model
    # # float16 --> bfloat16
    # short_model = short_model.to(torch.bfloat16)
    # lm_model = LLAMA(model=short_model, tokenizer=tokenizer, source_len=256, optim = optim)
    # lm_model.train(train_filename='./dataset/stack_filtered.csv', train_batch_size=1, learning_rate=1e-5, 
    #             num_train_epochs=1, do_eval=True, eval_filename='./dataset/code_generation/HumanEval.jsonl', 
    #             eval_batch_size=1, output_dir='./save_models/CodeQwen1.5-'+optim+'-0')
    # tokenizer.save_pretrained('./save_models/CodeQwen1.5-'+optim+'-0')
    # del short_model
    # del lm_model
    for i in range(n_layer-1):
        model, layer_map = ShortTransformer.from_pretrained('./save_models/CodeQwen1.5-'+optim+'-'+str(i), 
                                            torch_dtype=torch.float16, 
                                            device_map="cuda")
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
        # float16 --> bfloat16
        short_model = short_model.to(torch.bfloat16)
        lm_model = LLAMA(model=short_model, tokenizer=tokenizer, source_len=256)
        lm_model.train(train_filename='./dataset/stack_filtered.csv', train_batch_size=1, learning_rate=1e-5, 
                    num_train_epochs=1, do_eval=True, eval_filename='./dataset/code_generation/HumanEval.jsonl', 
                    eval_batch_size=1, output_dir='./save_models/CodeQwen1.5-'+optim+'-'+str(i+1))
        tokenizer.save_pretrained('./save_models/CodeQwen1.5-'+optim+'-'+str(i+1))
        del model
        del lm_model
    short_model = lm_model.get_model()
    return short_model, tokenizer

if __name__ == '__main__':
    optim = 'badam'
    model, tokenizer = greedy_prune(model_name='./save_models/CodeQwen1.5-7B-Chat-vocab', n_layer=8, optim=optim)
    tokenizer.save_pretrained("./save_models/CodeQwen1.5-7B-Chat-layer8-"+optim)
    model.config.save_pretrained("./save_models/CodeQwen1.5-7B-Chat-layer8-"+optim)
    model.save_pretrained("./save_models/CodeQwen1.5-7B-Chat-layer8-"+optim)
    