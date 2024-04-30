import pandas as pd
import matplotlib.pyplot as plt

all_active_ligands = pd.read_csv('../all_active_ligands.csv')
new_ligands = pd.read_csv('../new_ligands.csv')
samples = pd.read_csv('../samples.csv')
columns = ['EGFR.csv', 'ER-beta.csv', 'HER2.csv', 'NFKB.csv', 'PR.csv']
colors = ['tab:blue', 'tab:purple', 'tab:blue', 'tab:olive', 'tab:brown']
title = ['EGFR', 'ER', 'HER2', 'NFKB', 'PR']

for column, color, title in zip(columns, colors, title):
    fig, ax = plt.subplots()
    ax.scatter(all_active_ligands['molecular weight'], all_active_ligands[column], color=color, alpha=0.1)
    ax.scatter(samples['molecular weight'], samples[column], color=color)
    ax.scatter(new_ligands['molecular weight'], new_ligands[column], color='tab:red', alpha=0.5)
    ax.set_xlabel('Molecular Weight')
    ax.set_ylabel('Binding Energy (Kcal/mol)')
    ax.set_title(title)
    plt.ylim(-30, 0)
    plt.xlim(0, 1000)
    fig.set_size_inches(8, 6)
    plt.savefig(title + '.png', dpi=300)
    plt.close()