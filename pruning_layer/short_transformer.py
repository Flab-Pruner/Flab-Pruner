from functools import partial, wraps
import copy
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from pruning_layer.dist import angular_distance_last_token
from pruning_layer.utils import get_best_pruning_start, get_logger
from pruning_layer.model_factory.remove_and_duplicate_with_merging import merge_layers

logger = get_logger("short-transformers", debug=True)


class Memory:
    def __init__(self, layer_count: int):
        self.examples_count: int = -1
        self.result = np.zeros((layer_count, layer_count))
        self.layers_outputs: dict = {}


class ShortTransformer(PreTrainedModel):
    @classmethod
    def from_model(cls, model, distance=angular_distance_last_token):
        cls = model
        cls.layer_count = len(cls.model.layers)
        cls.distance = distance
        # add memory for storing intermediate layers outputs
        cls.memory = Memory(cls.layer_count)

        # @TODO add distances here
        # @TODO auto assign all methods from the class here
        cls.clear_memory = partial(ShortTransformer.clear_memory, cls)
        cls.analyse_layers = partial(ShortTransformer.analyse_layers, cls)
        cls.prune = partial(ShortTransformer.prune, cls)
        cls.remove_layers = partial(ShortTransformer.remove_layers, cls)
        cls.set_metric = partial(ShortTransformer.set_metric, cls)
        layer_map = {}
        # # add decorators to each forward in layers
        cls_cpu = cls.to("cpu")
        for layer_idx, layer in enumerate(cls_cpu.model.layers):
            layer_map[layer_idx] = copy.copy(layer)
        cls = cls.to("cuda")
        for layer_idx, layer in enumerate(cls.model.layers):
            layer.forward = ShortTransformer._layer_io(cls, layer_idx)(layer.forward)

        return cls, layer_map

    @classmethod
    def from_pretrained(cls, *args, **kw):
        # @TODO: support other AutoModels variants
        model = AutoModelForCausalLM.from_pretrained(*args, **kw)
        return cls.from_model(model)

    @staticmethod
    def clear_memory(model) -> None:
        model.memory = Memory(model.layer_count)
        
    @staticmethod
    def _layer_io(model, layer_idx: int):
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kw):

                input_hidden_states = args[0]
                # print(f"Layer {layer_idx} input shape: {input_hidden_states.shape}")
                if layer_idx == 0:
                    # clear the memory of previous example outputs and remmeber the input
                    model.memory.layers_outputs = {
                        -1: torch.clone(input_hidden_states).to("cpu")
                    }
                    model.memory.examples_count += 1
                # pass all arguments to the function
                
                result = f(*args, **kw)

                # calculate io metric for all layers:
                output_hidden_states = torch.clone(result[0]).to("cpu")

                # calculate scores from -1 to this layer:
                for k, v in model.memory.layers_outputs.items():
                    # print(f"Calculating distance layer {k}")
                    dist = model.distance(v, output_hidden_states)

                    cut_layers = layer_idx - k - 1
                    # print(f"Cut layers: {cut_layers}")
                    model.memory.result[cut_layers, k + 1] = (
                            model.memory.result[cut_layers, k + 1]
                            * model.memory.examples_count
                            + dist
                        ) / (model.memory.examples_count + 1)

                # remember the state
                model.memory.layers_outputs[layer_idx] = torch.clone(
                    output_hidden_states
                ).to("cpu")
                return result

            return wrap

        return decorator

    @staticmethod
    def set_metric(model, criterion_callable):
        model.distance = criterion_callable

    @staticmethod
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    @staticmethod
    def analyse_layers(
        model,
        data_list,
        tokenizer=None,
        use_chat_template=False,
        max_length: int = 1000
    ) -> None:
        if tokenizer is None:
            logger.debug(
                "Tokenizer not provided, will load tokenizer from config._name_or_path"
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            except Exception as e:
                logger.error(
                    f"Loading the tokenizer failed wwth error: {e}.\nUse analyse_layers(... tokenizer=...) to manually set the tokenizer."
                )
                raise RuntimeError

        model.model.eval()
        with torch.no_grad():
            for content in tqdm(data_list):
                if use_chat_template:
                    inputs = tokenizer.apply_chat_template(content, tokenize=True, add_generation_prompt=False)
                else:
                    inputs = tokenizer(
                        content,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=max_length,
                    ).to(model.device)
                model(inputs['input_ids'])
        result = model.memory.result
        model.clear_memory()
        return result

    @staticmethod
    def prune(model, layer_map, start_layer: int, block_size: int):
        for layer_idx, layer in enumerate(model.model.layers):
            model.model.layers[layer_idx] = layer_map[layer_idx].cuda()
        model = model.to("cuda")
        remove_layers = list(range(start_layer, start_layer + block_size))
        logger.debug(f"Removing layers: {remove_layers}")

        count = 0
        layer_count = model.layer_count
        if block_size == 1:
            del model.model.layers[remove_layers[0]]
            layer_count = layer_count-1
            for i in range(0, layer_count):
                layer = model.model.layers[i]
                layer.layer_idx = i
                layer.self_attn.layer_idx = i
        else:
            new_layers = torch.nn.ModuleList()
            for i in range(0, layer_count):
                if i not in remove_layers:
                    count += 1
                    layer = model.model.layers[i]
                    layer.layer_idx = count
                    layer.self_attn.layer_idx = count
                    new_layers.append(layer)
                    del model.model.layers[i]
            model.model.layers = new_layers
            del new_layers
        changed_num_hidden_layers = model.layer_count - block_size
        changed_model_name_or_path = (
            f"{model.config._name_or_path}-{changed_num_hidden_layers}L"
        )

        logger.debug(f"""Changing model config to reflect changes:
        config.num_hidden_layers: {model.config.num_hidden_layers} -> {changed_num_hidden_layers}
        config._name_or_path: {model.config._name_or_path} -> {changed_model_name_or_path}""")

        model.config.num_hidden_layers = changed_num_hidden_layers
        model.config._name_or_path = changed_model_name_or_path

        return model
    
    @staticmethod
    def prune_then_merge(model, layer_map, start_layer: int):
        for layer_idx, layer in enumerate(model.model.layers):
            model.model.layers[layer_idx] = layer_map[layer_idx].cuda()
        model = model.to("cuda")
        logger.debug(f"Removing layers: {start_layer}")
        layer_count = model.layer_count
        # merge 
        model.model.layers[start_layer+1] = merge_layers(model.model.layers[start_layer], model.model.layers[start_layer+1])
        del model.model.layers[start_layer]
        layer_count = layer_count-1
        for i in range(0, layer_count):
            layer = model.model.layers[i]
            layer.layer_idx = i
            layer.self_attn.layer_idx = i
        changed_num_hidden_layers = model.layer_count - 1
        changed_model_name_or_path = (
            f"{model.config._name_or_path}-{changed_num_hidden_layers}L"
        )

        logger.debug(f"""Changing model config to reflect changes:
        config.num_hidden_layers: {model.config.num_hidden_layers} -> {changed_num_hidden_layers}
        config._name_or_path: {model.config._name_or_path} -> {changed_model_name_or_path}""")

        model.config.num_hidden_layers = changed_num_hidden_layers
        model.config._name_or_path = changed_model_name_or_path

        return model
    
    @staticmethod
    def remove_layers(
        model,
        tokenizer,
        block_size,
        data_list,
        max_length=1000
    ):
        result = model.analyse_layers(
            data_list=data_list,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        logger.debug(f"Choosing optimal {block_size}-layers block to prune.")
        start_layer = get_best_pruning_start(result=result, block_size=block_size)
        logger.debug(f"Best 5-layers block to prune starts at layer: {start_layer}.")
        return model.prune(start_layer=start_layer, block_size=block_size)
