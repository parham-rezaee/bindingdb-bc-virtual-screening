import pandas as pd
import glob

file_list = glob.glob('results_*.tsv')

for file_path in file_list:
    print(f"File: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    counts = df['Target'].value_counts()
    for target, count in counts.items():
        print(f"Count of {target}: {count}")
    print() 