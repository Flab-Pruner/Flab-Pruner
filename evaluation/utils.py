from nlgeval import NLGEval
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
import openai
from tqdm import tqdm

openai.api_base = 'https://api.openai-proxy.org/v1'
openai.api_key = '--'

def prompt_gen(code):
    demo = '''# Instruct: Suppose you are a professional computer programming expert.
Please understand the given code, then determine if it is secure enough, i.e. if there are any security vulnerabilities.

If it is secure, only return "Security", otherwise only return "Insecure".

# Code:
'''
    return demo + code

def generate_by_openai(code, model):
    try:
        prompt = prompt_gen(code)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "If it is secure, only return 'Security', otherwise only return 'Insecure'. Do not give any reasons. \nReturn:"},
            ],
            temperature=0
        )
        result = response['choices'][0]['message']['content']
    except:
        result = 'nan'
    print(result)
    return result

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
    # print(results)
    return results

def calculate_bleu(references, hypothesis):
    nlgeval = NLGEval()  
    return nlgeval.compute_metrics([references], hypothesis)['Bleu_4']

def calculate_pass1(ref_file, hyp_file):
    result = entry_point(hyp_file, k='1', problem_file=ref_file)
    return result

def calculate_em(references, hypothesis):
    count = 0
    for i in range(len(hypothesis)):
        hyp=str(hypothesis[i]).replace("\"", "'")
        ref=str(references[i]).replace("\"", "'")
        if hyp == ref:
            count += 1
    return count/len(hypothesis)*100

def calculate_security(references):
    count = 0
    for code in tqdm(references):
        result = generate_by_openai(code, model='gpt-3.5-turbo')
        if result == 'Security':
            count += 1
    print(count/len(references)*100)