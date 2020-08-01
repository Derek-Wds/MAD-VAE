import json
import numpy as np

if __name__ == "__main__":
    models = {'vanilla': 0, 'classification': 0, 'proxi_dist': 0, 'combined': 0}
    models_list = ['vanilla', 'classification', 'proxi_dist', 'combined'] # for consistency in older versions

    for flavor in models_list:
        with open(f'./accuracy_{flavor}.txt', 'r') as f:
            models[flavor] = json.load(f)

    # Models initialized with their base accuracy
    acc = {'fgsm': [0.1316], 'r-fgsm': [0.1521], 'cw': [0.0075], 'mi-fgsm': [0.0074], 'pgd': [0.0073], 'single': [0.9977]}
    acc_name = {'fgsm': 'FGSM', 'r-fgsm': 'Rand-FGSM', 'cw': 'CW', 'mi-fgsm': 'MI-FGSM', 'pgd': 'PGD', 'single': 'Single Pixel'}
    acc_list = list(acc.keys())

    for model in acc_list:
        for flavor in models_list:
            acc[model].append(models[flavor][model])
        
        argmax = np.argmax(acc[model][1:]) + 1
        acc[model][argmax] = f'\\textbf{{{acc[model][argmax]}}}'
        
    with open('./whitebox_table.tex', 'w') as f:
        c = ['c'] * (len(models_list) + 3)
        f.write("\\begin{table}[H]\n\centering\n\\begin{tabular}{")
        f.write('|'.join(c))
        f.write("}\nAttack & No Attack & No Defense & Vanilla & Classification & Proximity and Distance & Combined \\\\ \\hline\n")
        for model in acc_list:
            acc[model].insert(0, 0.9931)
            acc[model].insert(0, acc_name[model])
            f.write(' & '.join(str(x) for x in acc[model]))
            f.write('\\\\\n')
        f.write('\\end{tabular}\n')
        f.write('\\caption{Classification accuracy of different models based on the FGSM, Rand-FGSM, CW, Momentum Iterative FGSM, PGD, and Single Pixel White-Box attack on the classifier with the default parameters. The models are trained on the data generated using the first three attack methods while the other three attacks are not included in the training dataset.}\n')
        f.write('\\label{table:whitebox-result}\n')
        f.write('\\end{table}\n')