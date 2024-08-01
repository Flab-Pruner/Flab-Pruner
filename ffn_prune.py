import torch

from hidden_prune_utils.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoConfig, AutoTokenizer

def prune(model_name, method):
    llama_config = AutoConfig.from_pretrained(model_name)
    prune_config = {
        "hidden_size_remain": 4096,
        "num_attention_heads_remain":32,
        "num_key_value_heads_remain":4,
        "ffn_hidden_size_remain":13184,
    }
    llama_config.update(prune_config)
    llama_model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llama_model.eval()
    params = sum(p.numel() for p in llama_model.parameters())
    print("Total params of original model: %.2fB" % (params / 1e9))

    llama_model.prune(config=llama_config, stage=method)
    new_params = sum(p.numel() for p in llama_model.parameters())
    print("Total params of pruned model: %.2fB" % (new_params / 1e9))
    new_config = AutoConfig.from_pretrained(model_name)
    new_config.hidden_size = llama_config.hidden_size_remain
    new_config.intermediate_size = llama_config.ffn_hidden_size_remain
    new_config.num_attention_heads = llama_config.num_attention_heads_remain
    new_config.num_key_value_heads = llama_config.num_key_value_heads_remain
    llama_model.config = new_config

    llama_model.save_pretrained(model_name+'_ffn')
    tokenizer.save_pretrained(model_name+'_ffn')

if __name__ == '__main__':
    prune('./save_models/AutoCoder_QW_7B_vocab_layer4', 'top')