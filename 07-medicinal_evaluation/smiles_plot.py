import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

df = pd.read_csv('smiles_list.csv')
df['smiles'] = df[df.columns[0]]
fig, ax = plt.subplots()

for index, row in df.iterrows():
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    ax.imshow(img)
    ax.axis('off')
    img.save(f'molecule_{index}.png', dpi=(300, 300))
    ax.cla()
    
plt.close(fig)