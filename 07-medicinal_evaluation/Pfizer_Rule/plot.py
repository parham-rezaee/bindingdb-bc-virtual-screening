import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bindingdb_df = pd.read_csv('../bindingdb_ligands.csv')
virtual_screening_df = pd.read_csv('../virtual_screening_ligands.csv')
samples_df = pd.read_csv('../samples.csv')
file_names = ['egfr+her2.smi', 'er.smi', 'nfkb.smi', 'pr.smi']
colors = ['tab:blue', 'tab:purple', 'tab:olive', 'tab:brown']
title = ['EGFR+HER2', 'ER', 'NFKB', 'PR']

for file_name, color, title in zip(file_names, colors,title):
    filtered_bindingdb_df = bindingdb_df[bindingdb_df['file name'] == file_name]
    filtered_virtual_screening_df = virtual_screening_df[virtual_screening_df['file name'] == file_name]
    filtered_samples_df = samples_df[samples_df['file name'] == file_name]
    plt.scatter(filtered_bindingdb_df['LogP'], filtered_bindingdb_df['TPSA'], color=color, alpha=0.1)
    plt.scatter(filtered_samples_df['LogP'], filtered_samples_df['TPSA'], color=color)
    plt.scatter(filtered_virtual_screening_df['LogP'], filtered_virtual_screening_df['TPSA'], color='tab:red', alpha=0.5)
    square_vertices = [(-3, -10), (3, -10), (3, 75), (15, 75), (15, 300), (-3, 300)]
    square_patch = plt.Polygon(square_vertices, color='green', alpha=0.1)
    plt.gca().add_patch(square_patch)
    plt.title(f'{title}')
    plt.xlabel('LogP')
    plt.ylabel('TPSA')
    plt.ylim(-10, 300)
    plt.xlim(-3, 12)
    plt.savefig(f'{title}.png', dpi=300)
    plt.clf()