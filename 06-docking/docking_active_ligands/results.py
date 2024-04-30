import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

smiles_dir = 'smiles'
docking_dir = 'docking_results'

df = pd.DataFrame(columns=['file name', 'name', 'smiles'])

for filename in os.listdir(smiles_dir):
    if filename.endswith('.smi'):
        file_path = os.path.join(smiles_dir, filename)
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    smiles, name = line.split('\t')
                    df = pd.concat([df, pd.DataFrame({'file name': [filename], 'name': [name], 'smiles': [smiles]})])

for filename in os.listdir(docking_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(docking_dir, filename)
        docking_df = pd.read_csv(file_path, header=None, names=['name', 'energy'])
        df = df.merge(docking_df, on='name', how='left')
        df.rename(columns={'energy': filename}, inplace=True)

df['molecular weight'] = df['smiles'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if x != '' and Chem.MolFromSmiles(x) is not None else -1)
samples_df = pd.read_csv('./samples/sample_active_target_molecules.csv', header=None, names=['name'])
filtered_df = df.merge(samples_df, on='name', how='inner')
virtual_df = pd.read_csv('./virtual_screening_result/results_a80_t90.tsv', delimiter='\t')
filtered_virtual_df = virtual_df[~virtual_df.name.isin(df.name)]
df.to_csv('all_active_ligands.csv', index=False)
filtered_df.to_csv('samples.csv', index=False)
filtered_virtual_df.to_csv('new_ligands_list.csv', index=False)

