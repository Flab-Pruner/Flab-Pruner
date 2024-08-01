import os
from peft import PeftModel
from nlp2 import set_seed
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

set_seed(42)

def generate_dec(code, tokenizer, model, device, max_len):
    STOP_SEQS = ['\nclass', '\ndef', '\nuser', '\n#', '\nif', '\nprint', "'''\n```python", "```"]
    BEGIN_SQE = '"""'
    code = code.strip()
    encoding = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        gen_tokens = model.generate(input_ids=encoding.input_ids,
                                    attention_mask=encoding.attention_mask,
                                    do_sample=False,
                                    use_cache=True, 
                                    max_new_tokens=max_len,
                                    num_return_sequences=1,
                                    pad_token_id=tokenizer.eos_token_id)
    gen_tokens = gen_tokens[:, encoding['input_ids'].shape[-1]:]
    completion_seq = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    for stop_seq in STOP_SEQS:
        index = completion_seq.find(stop_seq)
        if index != -1:
            completion_seq = completion_seq[:index]
    begin_index = completion_seq.find(BEGIN_SQE)
    if begin_index != -1:
        completion_seq = completion_seq[begin_index + len(BEGIN_SQE):]
    print(completion_seq)
    return completion_seq

def main(dataset, model, method):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model)
    lm_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    lm_model.to(device)
    # lm_model = AutoModelForCausalLM.from_pretrained(
    #     model,
    #     load_in_4bit=True, device_map="auto")

    lm_model.eval()

    if dataset == 'humaneval':
        problems = read_problems('./dataset/code_generation/HumanEval.jsonl')
    if dataset == 'openeval':
        problems = read_problems('./dataset/code_generation/OpenEval_format.jsonl')
    if dataset == 'mbpp':
        problems = read_problems('./dataset/code_generation/MBPP_test.jsonl')
    if dataset == 'codeharmony':
        problems = read_problems('./dataset/code_generation/CodeHarmony_test.jsonl')
    if dataset == 'mhpp':
        problems = read_problems('./dataset/code_generation/MHPP_format.jsonl')
    
    if dataset == 'odex':
        problems = read_problems('./dataset/code_generation/ODEX_format.jsonl')
    if dataset == 'tool_use':
        problems = read_problems('./dataset/others/tool_use.jsonl')
    if dataset == 'combine':
        problems = read_problems('./dataset/others/combine.jsonl')
    if dataset == 'subtle':
        problems = read_problems('./dataset/others/subtle.jsonl')
    if dataset == 'creative':
        problems = read_problems('./dataset/others/creative.jsonl')
    if dataset == 'difficult':
        problems = read_problems('./dataset/others/difficult.jsonl')
    # from codecarbon import EmissionsTracker
    # tracker = EmissionsTracker()
    # tracker.start()
    samples = []
    for task_id in tqdm(problems):
        prompt = problems[task_id]["prompt"]
        completion_seq = generate_dec(prompt, tokenizer, lm_model, device, max_len = 256)
        samples.append(dict(task_id=task_id, completion=completion_seq))
    if not os.path.exists('results/code_generation/'+method+'/'+dataset):
        os.makedirs('results/code_generation/'+method+'/'+dataset)
    write_jsonl('results/code_generation/'+method+'/'+dataset+'/AutoCoder.jsonl', samples)
    # emissions: float = tracker.stop()
    # print(emissions)

if __name__ == '__main__':
    models = ['./save_models/AutoCoder_instruct']
    datasets = ['humaneval']
    methods = ['instruct']
    for model in models:
        for dataset in datasets:
            for method in methods:
                main(dataset, model, method)
    

