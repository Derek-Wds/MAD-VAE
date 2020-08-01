import json
import numpy as np

if __name__ == "__main__":
    models = {'identity': 0, 'vanilla': 0, 'classification': 0, 'proxi_dist': 0, 'combined': 0}
    models_list = ['identity', 'vanilla', 'classification', 'proxi_dist', 'combined']# for consistency in older versions

    for flavor in models_list:
        with open(f'./results/accuracy_{flavor}.txt', 'r') as f:
            models[flavor] = json.load(f)

    # Models initialized with their base accuracy
    attacks = {'OP': ['OP'], 'OP In-Bounds': ['OP In-Bounds'], 'Total': ['Total'], 'Total In-Bounds': ['Total In-Bounds']}
    for flavor in models_list:
        attacks['OP'].append(models[flavor]['all'][0])
        attacks['OP In-Bounds'].append(models[flavor]['all'][1])
        attacks['Total'].append(models[flavor]['all'][2])
        attacks['Total In-Bounds'].append(models[flavor]['all'][3])
    
    for attack in attacks:
        argmax = np.argmax(attacks[attack][1:]) + 1
        attacks[attack][1:] = [round(x, 4) for x in attacks[attack][1:]]
        attacks[attack][argmax] = f'\\textbf{{{attacks[attack][argmax]}}}'

    with open('./OP_table.tex', 'w') as f:
        c = ['c'] * (len(models_list) + 1)
        f.write("\\begin{table}[H]\n\centering\n\\begin{tabular}{")
        f.write('|'.join(c))
        f.write("}\nAttack & Identity & Vanilla & Classification & Proximity and Distance & Combined \\\\ \\hline\n")
        f.write(' & '.join(str(x) for x in attacks['OP']))
        f.write('\\\\\n')
        f.write(' & '.join(str(x) for x in attacks['OP In-Bounds']))
        f.write('\\\\\n')
        f.write(' & '.join(str(x) for x in attacks['Total']))
        f.write('\\\\\n')
        f.write(' & '.join(str(int(x)) for x in attacks['Total In-Bounds']))
        f.write('\\\\\n')
        f.write('\\end{tabular}\n')
        f.write('\\caption{Classification accuracy of different defensive models under the Overpowered Attack as well as the number of samples for both general and in-bounds (under budget) attacks.}\n')
        f.write('\\label{table:whitebox-result}\n')
        f.write('\\end{table}\n')