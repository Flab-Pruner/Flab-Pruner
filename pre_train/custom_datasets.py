import copy

import pandas as pd

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GPTDataset(Dataset):
    def __init__(self, datafile, tokenizer, source_len=256, cutoff_len=512):

        self.cutoff_len = cutoff_len
        self.source_len = source_len

        self.inputs = []
        self.token_labels = []
        if '.csv' in datafile:
            datas = pd.read_csv(datafile)
        elif '.jsonl' in datafile:
            datas = pd.read_json(datafile, lines=True)

        length = len(datas)
        # length = 100

        for idx in tqdm(range(length)):
            if '.csv' in datafile:
                src = datas["src"][idx]
                tgt = datas["tgt"][idx]
            elif '.jsonl' in datafile:
                src = datas["prompt"][idx]
                tgt = datas["canonical_solution"][idx]
            try:
                input_ids, input_labels = self.tokenize_prompt(src, tgt, tokenizer, source_len, cutoff_len)
                self.inputs.append(input_ids)
                self.token_labels.append(input_labels)
            except:
                pass
#               print("Error in tokenizing prompt")

    def tokenize(self, prompt, tokenizer, cutoff_len, padding=False):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding='max_length' if padding else False,
            return_tensors=None
        )
        return {
            "input_ids": result["input_ids"],
            "labels": copy.deepcopy(result["input_ids"])
        }

    def tokenize_prompt(self, src, tgt, tokenizer, raw_source_len, cutoff_len):
        # 输入的分词， 输入的最大长度为256
        # tokenized_result = self.src_tokenize(src, tokenizer, cutoff_len)

        tokenized_result = self.tokenize(src, tokenizer, raw_source_len, padding=False)

        source_len = len(tokenized_result['input_ids'])

        assert source_len<=raw_source_len
        assert len(tgt)>0

        src = tokenizer.decode(tokenized_result['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # 输入+输出
        prompt_with_response = src + tgt + tokenizer.eos_token

        # 输入+输出 的分词
        tokenized_with_response = self.tokenize(prompt_with_response, tokenizer, cutoff_len, padding=True)

        tokenized_with_response["labels"] = [-100] * source_len + tokenized_with_response["labels"][source_len:]
        # print(tokenizer.decode(tokenized_with_response["labels"], skip_special_tokens=True,
        #                        clean_up_tokenization_spaces=True))

        assert len(tokenized_with_response["input_ids"]) == len(tokenized_with_response["labels"])

        return tokenized_with_response["input_ids"], tokenized_with_response["labels"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])



class Dataset_Pretrain(Dataset):
    def __init__(self, datafile, tokenizer, source_len=512):

        self.source_len = source_len

        self.inputs = []
        # self.token_labels = []
        if '.csv' in datafile:
            df = pd.read_csv(datafile)
            data_list = df['content'].tolist()
        elif '.jsonl' in datafile:
            datas = pd.read_json(datafile, lines=True)
            srcs = datas["prompt"].tolist()
            tgts = datas["canonical_solution"].tolist()
            data_list = [src+tgt for src, tgt in zip(srcs, tgts)]
        length = len(data_list)

        for idx in tqdm(range(length)):
            src_code = data_list[idx].strip()
            input_ids = tokenizer(src_code, 
                                  truncation=True,
                                  max_length=source_len,
                                  padding='max_length',
                                  return_tensors=None)['input_ids']
            self.inputs.append(input_ids)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.inputs[item])