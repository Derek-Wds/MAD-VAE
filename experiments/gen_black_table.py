import json
import numpy as np

if __name__ == "__main__":
    models = {'vanilla': 0, 'classification': 0, 'proxi_dist': 0, 'combined': 0}
    models_list = ['vanilla', 'classification', 'proxi_dist', 'combined']# for consistency in older versions

    for flavor in models_list:
        with open(f'./accuracy_{flavor}_black.txt', 'r') as f:
            models[flavor] = json.load(f)

    # Models initialized with their base accuracy
    acc = {'a': [0.993900], 'b': [0.992500], 'c': [0.993800], 'd': [0.981200], 'e': [0.980700]}
    acc_list = list(acc.keys())

    for model in acc_list:
        acc[model].append(models['vanilla'][f'{model}-fgsm-attack'])
        for flavor in models_list:
            acc[model].append(models[flavor][f'{model}-fgsm'])
        
        argmax = np.argmax(acc[model][1:]) + 1
        acc[model][argmax] = f'\\textbf{{{acc[model][argmax]}}}'
        
    with open('./blackbox_table.tex', 'w') as f:
        c = ['c'] * (len(models_list) + 3)
        f.write("\\begin{table}[H]\n\centering\n\\begin{tabular}{")
        f.write('|'.join(c))
        f.write("}\nSubstitute & No Attack & No Defense & Vanilla & Classification & Proximity and Distance & Combined \\\\ \\hline\n")
        for model in acc_list:
            acc[model].insert(0, model.upper())
            f.write(' & '.join(str(x) for x in acc[model]))
            f.write('\\\\\n')
        f.write('\\end{tabular}\n')
        f.write('\\caption{Classification accuracy of different models based on the FGSM Black-Box attack on various substitute models with $\epsilon=0.3$.}\n')
        f.write('\\label{table:blackbox-result}\n')
        f.write('\\end{table}\n')