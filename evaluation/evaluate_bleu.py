from nlgeval import NLGEval
import pandas as pd

nlgeval = NLGEval()  

datasets = ['codeharmony']
method = 'instruct'
models = ['']

for model in models:
    for dataset in datasets:
        if dataset == 'humaneval':
            problems_df = pd.read_csv('./dataset/cot_generation/humaneval_cot.csv')
        if dataset == 'openeval':
            problems_df = pd.read_csv('./dataset/cot_generation/openeval_cot.csv')
        if dataset == 'codecot':
            problems_df = pd.read_csv('./dataset/cot_generation/CodeCoT_test.csv')
        if dataset == 'codeharmony':
            problems_df = pd.read_csv('./dataset/cot_generation/CodeHarmony_test.csv')
        references = problems_df['cot'].tolist()
        df = pd.read_csv('./int4/cot_generation/'+method+'/'+dataset+'/Nxcode.csv', header=None)
        hypothesis = df[0].tolist()
        metrics_dict = nlgeval.compute_metrics([references], hypothesis)
        print(dataset, model)
        print(metrics_dict['Bleu_4'])
        