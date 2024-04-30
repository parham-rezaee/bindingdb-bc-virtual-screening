import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

bindingdb_df = pd.read_csv('../bindingdb_ligands.csv')
virtual_screening_df = pd.read_csv('../virtual_screening_ligands.csv')
samples_df = pd.read_csv('../samples.csv')
file_names = ['egfr+her2.smi', 'er.smi', 'nfkb.smi', 'pr.smi']
colors = ['tab:blue', 'tab:purple', 'tab:olive', 'tab:brown']
titles = ['EGFR+HER2', 'ER', 'NFKB', 'PR']

for file_name, color, title in zip(file_names, colors, titles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    filtered_bindingdb_df = bindingdb_df[bindingdb_df['file name'] == file_name]
    filtered_virtual_screening_df = virtual_screening_df[virtual_screening_df['file name'] == file_name]
    filtered_samples_df = samples_df[samples_df['file name'] == file_name]
    ax.scatter(filtered_bindingdb_df['QED'], filtered_bindingdb_df['MCE-18'], filtered_bindingdb_df['Synth'], color=color, alpha=0.1)
    ax.scatter(filtered_samples_df['QED'], filtered_samples_df['MCE-18'], filtered_samples_df['Synth'], color=color)
    ax.scatter(filtered_virtual_screening_df['QED'], filtered_virtual_screening_df['MCE-18'], filtered_virtual_screening_df['Synth'], color='tab:red', alpha=0.5)
    vertices = [
        [0.67, 45, 0], [1, 45, 0], [1, 250, 0], [0.67, 250, 0],
        [0.67, 45, 6], [1, 45, 6], [1, 250, 6], [0.67, 250, 6] 
    ]
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]

    cubic = Poly3DCollection([[vertices[index] for index in face] for face in faces], alpha=0.1, edgecolor='black', linewidths=0.5)
    cubic.set_facecolor('green')
    ax.add_collection3d(cubic)
    ax.set_title(title)
    ax.set_xlabel('QED')
    ax.set_ylabel('MCE-18')
    ax.set_zlabel('SAscore')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 250)
    ax.set_zlim(0, 10)
    plt.savefig(f'{title}.png', dpi=300)
    plt.clf()