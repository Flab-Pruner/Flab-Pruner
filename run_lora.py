from nlp2 import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from pre_train.LLAMA_model_lora import LLAMA
import torch
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)
set_seed(42)

def train(model):
    tokenizer = AutoTokenizer.from_pretrained(
        model)
    lm_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    lm_model = LLAMA(model=lm_model, tokenizer=tokenizer, 
                load_adapter_path="None", source_len=256, cutoff_len=512)

    lm_model.train(train_filename='./dataset/Instruct_Dataset.csv', train_batch_size=2, learning_rate=3e-4, 
                        num_train_epochs=1, do_eval=True, eval_filename='./dataset/code_generation/HumanEval.jsonl', 
                        eval_batch_size=1, output_dir='./save_models/AutoCoder-instruct')

# train('CodeQwen1.5-5.58B')
train('./save_models/AutoCoder_QW_7B_vocab_layer4_ffn')