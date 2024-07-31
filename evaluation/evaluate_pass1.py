from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    return results


if __name__ == '__main__':
    # entry_point('./results/code_generation/beam_pruned/openeval/CodeQwen1.5-5.6B-pruned.jsonl', k='1',
    #                     problem_file='./dataset/code_generation/OpenEval_format.jsonl')
    method = 'instruct'
    models = ['CodeQwen1.5-7B-Chat']
    # datasets = ['humaneval', 'openeval', 'mhpp', 'codeharmony']
    datasets = ['odex', 'tool_use', 'combine', 'subtle', 'creative', 'difficult']
    results = {}
    for model in models:
        for dataset in datasets:
            if dataset == 'humaneval':
                problems = './dataset/code_generation/HumanEval.jsonl'
            if dataset == 'openeval':
                problems = './dataset/code_generation/OpenEval_format.jsonl'
            if dataset == 'mbpp':
                problems = './dataset/code_generation/MBPP_test.jsonl'
            if dataset == 'codeharmony':
                problems = './dataset/code_generation/CodeHarmony_test.jsonl'
            if dataset == 'mhpp':
                problems = './dataset/code_generation/MHPP_format.jsonl'
            if dataset == 'odex':
                problems = './dataset/code_generation/ODEX_format.jsonl'
            if dataset == 'tool_use':
                problems = './dataset/others/tool_use.jsonl'
            if dataset == 'combine':
                problems = './dataset/others/combine.jsonl'
            if dataset == 'subtle':
                problems = './dataset/others/subtle.jsonl'
            if dataset == 'creative':
                problems = './dataset/others/creative.jsonl'
            if dataset == 'difficult':
                problems = './dataset/others/difficult.jsonl'
            result = entry_point('./results/others/'+method+'/'+dataset+'/Nxcode.jsonl', k='1',
                        problem_file=problems)
            results[dataset] = result
    print('-----------')
    print(results)
    