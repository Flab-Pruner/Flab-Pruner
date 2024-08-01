import torch

from modeling_llama import LlamaForCausalLM
from transformers import AutoConfig

llama_config = AutoConfig.from_pretrained("/media/yg/E/models/deepseek-coder-6.7b-instruct")
prune_config = {
    "hidden_size_remain": 3968,
    "num_attention_heads_remain":31,
    "ffn_hidden_size_remain":13184
}
llama_config.update(prune_config)
llama_model = LlamaForCausalLM.from_pretrained("/media/yg/E/models/deepseek-coder-6.7b-instruct", torch_dtype=torch.bfloat16)
llama_model.eval()
params = sum(p.numel() for p in llama_model.parameters())
print("Total params of original model: %.2fB" % (params / 1e9))

llama_model.prune(config=llama_config, stage='top')
new_params = sum(p.numel() for p in llama_model.parameters())
print("Total params of pruned model: %.2fB" % (new_params / 1e9))
new_config = AutoConfig.from_pretrained("/media/yg/E/models/deepseek-coder-6.7b-instruct")
new_config.hidden_size = llama_config.hidden_size_remain
new_config.intermediate_size = llama_config.ffn_hidden_size_remain
new_config.num_attention_heads = llama_config.num_attention_heads_remain
new_config.num_key_value_heads = llama_config.num_attention_heads_remain
llama_model.config = new_config

llama_model.save_pretrained('/media/yg/E/models/deepseek-coder-6.7b-instruct_top')
