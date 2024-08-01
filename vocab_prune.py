import json
import os
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

def load_datasets(dataset_files):
    # 加载数据集，可自行配置
    texts = []
    for dataset_file in dataset_files:
        df = pd.read_csv(dataset_file)
        srcs = df['src'].tolist()
        tgts = df['tgt'].tolist()
        texts.extend(srcs)
        texts.extend(tgts)
    return texts

def vocab_prune(dataset_files, raw_model_path, save_vocab=False, save_path=None):
    texts = load_datasets(dataset_files)
    print(f"Reading dataset length: {len(texts)}.")
    if save_path is None:
        save_path = raw_model_path + "-vocab-pruned"
    print(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # # 加载原始分词器和模型
    model = AutoModelForCausalLM.from_pretrained(raw_model_path,
                                                torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(raw_model_path)
    # print(model)
    # 记录在数据集中出现过的token（在原始分词器中）
    tokens = []
    for line in tqdm(texts):
        ids = tokenizer.encode(str(line))
        for id in ids:
            token = tokenizer.convert_ids_to_tokens(id)
            tokens.append(token)
    tokens.append(tokenizer.eos_token)
    tokens.append(tokenizer.pad_token)
    tokens.append(tokenizer.bos_token)
    tokens.append(tokenizer.unk_token)
    
    # 加载原始分词器中的vocabulary以及merges
    with open(raw_model_path+"/tokenizer.json", "r", encoding="utf-8") as f:
        file = json.load(f)
        vocabs = file["model"]["vocab"]
        added_tokens = file["added_tokens"]
        for add_token in added_tokens:
            vocabs[add_token["content"]] = add_token["id"]
            tokens.append(add_token["content"])
        merges = file["model"]["merges"]

    tokens = list(set(tokens))
    # token2id
    token_dict = {}
    idx = 0
    for token in tokens:
        token_dict[token] = idx
        idx += 1
    if save_vocab:
        # 临时保存为vocab.json，并手动替换
        pruned_vocab_file = save_path+'/vocab.json'
        with open(pruned_vocab_file, 'w', encoding='utf-8') as f:
            json.dump(token_dict, f, ensure_ascii=False, indent=8)
    print(f"New embedding size {len(tokens)}. Reintialize the tokenizer!")

    # 合并merges
    new_merges = []
    for merge in tqdm(merges):
        if merge.split(" ")[0] in list(token_dict.keys()) and merge.split(" ")[1] in list(token_dict.keys()) \
                and merge.split(" ")[0] + merge.split(" ")[1] in list(token_dict.keys()):
            new_merges.append(merge)
    print('Before Merge: '+ str(len(merges)))
    print('After Merge: '+ str(len(new_merges)))

    file["model"]["vocab"] = token_dict
    file["model"]["merges"] = new_merges
    with open(save_path+'/tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(file, f, indent=4, ensure_ascii=False)
    shutil.copy(raw_model_path+"/tokenizer_config.json", save_path)
    print("tokenizer update done!")

    # 创建新的embed_tokens和lm_head，用于权重更新
    vocab_size = len(tokens)
    # vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
    new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False, dtype=model.dtype)
    for token in tqdm(tokens):
        new_embeds.weight.data[token_dict[token]] = model.model.embed_tokens.weight.data[vocabs[token]]
        new_lm_head.weight.data[token_dict[token]] = model.lm_head.weight.data[vocabs[token]]
    model.model.embed_tokens.weight = new_embeds.weight
    model.lm_head.weight = new_lm_head.weight
    print("model update done!")
    print(model)
    # 保持模型的权重和配置文件
    model.config.vocab_size = vocab_size
    model.config.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print("model save done!")

def analysis(raw_model_path, pruned_model_path=None):
    model = AutoModelForCausalLM.from_pretrained(raw_model_path,
                                                torch_dtype=torch.bfloat16, device_map='cuda')
    old_tokenizer = AutoTokenizer.from_pretrained(raw_model_path)
    old_params = sum(p.numel() for p in model.parameters())
    print("Total params of original model: %.2fB" % (old_params / 1e9))
    del model

    model = AutoModelForCausalLM.from_pretrained(pruned_model_path,
                                                torch_dtype=torch.bfloat16, device_map='cuda')
    new_tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)
    new_params = sum(p.numel() for p in model.parameters())
    print("Total params of pruned model: %.2fB" % (new_params / 1e9))

    print('词表缩小为原来的:{}%'.format(round(len(new_tokenizer) / len(old_tokenizer), 4) * 100))
    print('模型参数量缩小为原来的:{}%'.format(round(new_params / old_params, 4) * 100))

if __name__ == '__main__':
    vocab_prune(dataset_files=['./dataset/Instruct_Dataset.csv'],
                raw_model_path='/media/yg/E/models/AutoCoder_QW_7B',
                save_vocab=False,
                save_path='./save_models/AutoCoder_QW_7B_vocab')
    analysis(raw_model_path='/media/yg/E/models/AutoCoder_QW_7B',
            pruned_model_path='./save_models/AutoCoder_QW_7B_vocab')