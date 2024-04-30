import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('0-features_a90_t90.tsv', delimiter='\t')
selected_columns = ['Eig01_AEA(dm)', 'RDF105p', 'RDF105s'] # 'HATS7v', 'MWC10', 'F03[C-C]', 'F09[N-Cl]', 'F03[C-N]', 'NsCl', 'MaxaasN', 'VE3sign_Dz(Z)', 'CATS2D_00_DA', 'NNRS', 'CATS3D_00_AA', 'mintsC', 'VE2_B(p)', 'SpMAD_G'
color_mapping = {1: 'tab:blue', 2: 'tab:purple', 3: 'tab:olive', 4: 'tab:brown'}

for i in range(len(selected_columns)):
    for j in range(i + 1, len(selected_columns)):
        for k in range(j + 1, len(selected_columns)):
            col_x = selected_columns[i]
            col_y = selected_columns[j]
            col_z = selected_columns[k]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            filtered_df = df[df['Target'].isin([2])]
            colors = [color_mapping[target] for target in filtered_df['Target']]
            ax.scatter(filtered_df[col_x], filtered_df[col_y], filtered_df[col_z], c=colors)
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_zlabel(col_z)
            plt.savefig(f'{col_x}_vs_{col_y}_vs_{col_z}.png', dpi=300)
            plt.close(fig)