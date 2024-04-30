import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bindingdb_df = pd.read_csv('../bindingdb_ligands.csv')
virtual_screening_df = pd.read_csv('../virtual_screening_ligands.csv')
samples_df = pd.read_csv('../samples.csv')
file_names = ['egfr+her2.smi', 'er.smi', 'nfkb.smi', 'pr.smi']
colors = ['tab:blue', 'tab:purple', 'tab:olive', 'tab:brown']
title = ['EGFR+HER2', 'ER', 'NFKB', 'PR']

# for file_name, color, title in zip(file_names, colors,title):
#     filtered_bindingdb_df = bindingdb_df[bindingdb_df['file name'] == file_name]
#     filtered_virtual_screening_df = virtual_screening_df[virtual_screening_df['file name'] == file_name]
#     filtered_samples_df = samples_df[samples_df['file name'] == file_name]
#     plt.scatter(filtered_bindingdb_df['nHA'], filtered_bindingdb_df['nHD'], color=color, alpha=0.1)
#     plt.scatter(filtered_samples_df['nHA'], filtered_samples_df['nHD'], color=color)
#     plt.scatter(filtered_virtual_screening_df['nHA'], filtered_virtual_screening_df['nHD'], color='tab:red', alpha=0.5)
#     square_vertices = [(0, 0), (0, 5), (10, 5), (10, 0)]
#     square_patch = plt.Polygon(square_vertices, color='green', alpha=0.1)
#     plt.gca().add_patch(square_patch)
#     plt.title(f'{title}')
#     plt.xlabel('Number of hydrogen bond acceptors')
#     plt.ylabel('Number of hydrogen bond donors')
#     plt.ylim(0, 15)
#     plt.xlim(0, 25)
#     plt.savefig(f'{title}_Hac_Hdo.png', dpi=300)
#     plt.clf()

for file_name, color, title in zip(file_names, colors,title):
    filtered_bindingdb_df = bindingdb_df[bindingdb_df['file name'] == file_name]
    filtered_virtual_screening_df = virtual_screening_df[virtual_screening_df['file name'] == file_name]
    filtered_samples_df = samples_df[samples_df['file name'] == file_name]
    plt.scatter(filtered_bindingdb_df['LogP'], filtered_bindingdb_df['MW'], color=color, alpha=0.1)
    plt.scatter(filtered_samples_df['LogP'], filtered_samples_df['MW'], color=color)
    plt.scatter(filtered_virtual_screening_df['LogP'], filtered_virtual_screening_df['MW'], color='tab:red', alpha=0.5)
    square_vertices = [(-3, 0), (-3, 500), (5, 500), (5, 0)]
    square_patch = plt.Polygon(square_vertices, color='green', alpha=0.1)
    plt.gca().add_patch(square_patch)
    plt.title(f'{title}')
    plt.xlabel('Log P')
    plt.ylabel('MW')
    plt.ylim(0, 1000)
    plt.xlim(-3, 12)
    plt.savefig(f'{title}_MW_LogP.png', dpi=300)
    plt.clf()