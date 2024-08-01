import math

import torch
import numpy as np

# The Unreasonable Ineffectiveness of the Deeper Layers
# https://arxiv.org/abs/2403.17887
# @TODO: make it work with batches
def get_angular_distance_ith_token(i: int) -> float:
    def angular_distance_ith_token(input, output) -> float:
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        try:
            input_last_hidden_state = input[:, i, :]
            output_last_hidden_state = output[:, i, :]
        except IndexError as e:
            print(
                f"{e}\nMake sure each sequence in the dataset no shorter than `{i}`-tokens."
            )
            raise RuntimeError

        sim = cos_sim(input_last_hidden_state, output_last_hidden_state)
        sim = torch.clamp(sim, -1.0, 1.0)
        dist = (1 / math.pi) * torch.acos(sim).item()
        return dist

    return angular_distance_ith_token


angular_distance_last_token = get_angular_distance_ith_token(-1)


def angular_distance_all_tokens(input, output):
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    sim = cos_sim(input, output)
    sim = torch.clamp(sim, -1.0, 1.0)

    dist = (1 / math.pi) * torch.acos(sim)
    dist = dist.mean().item()
    return dist

# Similar to: Your Transformer is Secretly Linear
# https://arxiv.org/pdf/2405.12250
def get_linear_approximation_ith_token(i: int) -> float:
    def linear_approximation_ith_token(input, output) -> float:
        try:
            input_last_hidden_state = input[:, i, :].numpy().astype(np.float32)
            output_last_hidden_state = output[:, i, :].numpy().astype(np.float32)
        except IndexError as e:
            print(
                f"{e}\nMake sure each sequence in the dataset no shorter than `{i}`-tokens."
            )
            raise RuntimeError

        A_est = np.linalg.pinv(input_last_hidden_state.T).dot(output_last_hidden_state.T).T
        output_est = A_est.dot(input_last_hidden_state)
        diff_squared = (output_est - output_last_hidden_state)**2
        diff_summ = diff_squared.sum()
        diff = diff_summ / diff_summ.size
        return np.log(diff)

    return linear_approximation_ith_token

# def linear_approximation_all_tokens(input, output):
#     A_est = np.linalg.pinv(input.T).dot(output.T).T
#     output_est = A_est.dot(input)
#     diff_squared = (output_est - output)**2
#     diff_summ = diff_squared.sum()
#     diff = diff_summ / diff_summ.size
#     return diff


def get_euclidian_dist_ith_token(i: int) -> float:
    def euclidian_dist_ith_token(input, output) -> float:
        try:
            input_last_hidden_state = input[:, i, :]
            output_last_hidden_state = output[:, i, :]
        except IndexError as e:
            print(
                f"{e}\nMake sure each sequence in the dataset no shorter than `{i}`-tokens."
            )
            raise RuntimeError

        dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(input_last_hidden_state, output_last_hidden_state), 2))) 
        return dist.item()

    return euclidian_dist_ith_token

# @TODO def get_euclidean_dist_all_tokens

# ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
# https://arxiv.org/pdf/2403.03853
def bi_score(input, output) -> float:
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim = cos_sim(input, output).mean().item()
    return 1 - sim

# Weight subcloning: direct initialization of transformers using larger pretrained ones
# https://arxiv.org/abs/2312.09299
def relative_magnitude(input, output, eps=1e-6) -> float:
    score = torch.norm(output, dim=-1)/(torch.norm(output + input, dim=-1) + eps)
    return score.mean().item()

def bi_score_last(input, output) -> float:
    # cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    # sim = cos_sim(input[:, -1, :].norm(dim=-1, keepdim=True), output[:, -1, :].norm(dim=-1, keepdim=True)).mean().item()
    # return 1 - sim
    input_hidden_state = input[:, -1, :]
    output_hidden_state = output[:, -1, :]
    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)

    sim = sim.clamp(min=-1.0, max=1.0)
    sim = sim.diagonal().nan_to_num(nan=-1.0)
    return 1 - sim
