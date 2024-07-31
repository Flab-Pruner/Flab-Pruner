import pandas as pd
# 'CodeQwen1.5-7B-Chat', 'Nxcode-CQ-7B-orpo'
models = ['']
method = 'instruct'
for dataset in ['codeharmony']:
    for model in models:
        df = pd.read_csv('./int4/code_output/'+method+'/'+dataset+'/Nxcode.csv')
        hypothesis = df['generated'].tolist()
        references = df['actual'].tolist()
        count = 0
        for i in range(len(hypothesis)):
            hyp=str(hypothesis[i]).replace("\"", "'")
            ref=str(references[i]).replace("\"", "'")
            if hyp == ref:
                count += 1
        print(model+dataset+': ', str(count/len(hypothesis)*100))
