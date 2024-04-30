import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('0-features_a90_t90.tsv', delimiter='\t')
selected_columns = ['CATS2D_00_DA','F03[C-N]','HATS7v','NNRS','CATS3D_00_AA','ATS6e','Eig01_AEA(dm)','mintsC','VE2_B(p)','SpMAD_G','L1i','S3K','MaxaasN','SHED_DD','Mor17v','CATS3D_03_LL','RDF105p','Mor26e','HyWi_Dz(e)','SM4_B(p)','Mor19e','MATS2v','F03[C-C]','Eig15_AEA(ed)','SM12_AEA(dm)','F09[N-Cl]','SM12_AEA(bo)','RDF020p','RDF105s','minssO','Mor23i','ATS6v','MWC10','VE3_B(m)','Eta_L_A','Eig01_EA','Mor10p','Eta_D_beta','L1s','VE3sign_Dz(Z)','Eig01_EA(dm)','ATS6m','P_VSA_e_1','Eig02_AEA(ed)','NsCl','ATSC8i','RDF065s','SM10_EA(ri)','Mor11p','VE1sign_B(p)','SM4_Dz(p)','TDB01m','RDF045s','Psi_e_0','SpAD_Dz(m)']
color_mapping = {1: 'tab:blue', 2: 'tab:purple', 3: 'tab:olive', 4: 'tab:brown'}

for i in range(len(selected_columns)):
    for j in range(i + 1, len(selected_columns)):
        col_x = selected_columns[i]
        col_y = selected_columns[j]
        fig, ax = plt.subplots()
        filtered_df = df[df['Target'].isin([2, 4])]
        colors = [color_mapping[target] for target in filtered_df['Target']]
        ax.scatter(filtered_df[col_x], filtered_df[col_y], c=colors)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        plt.savefig(f'{col_x}_vs_{col_y}.png', dpi=300)
        plt.close(fig)