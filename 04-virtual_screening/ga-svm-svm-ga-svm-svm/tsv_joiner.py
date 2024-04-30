import pandas as pd
import glob

directory = "./results/"
tsv_files = glob.glob(directory + "*_a75_t90.tsv")
combined_data = pd.DataFrame()

for file in tsv_files:
    data = pd.read_csv(file, delimiter="\t")
    combined_data = pd.concat([combined_data, data], ignore_index=True)

output_file = "./results_a75_t90.tsv"
combined_data.to_csv(output_file, sep="\t", index=False)

#########################

directory = "./features/"
tsv_files = glob.glob(directory + "*_a75_t90.tsv")
combined_data = pd.DataFrame()

for file in tsv_files:
    data = pd.read_csv(file, delimiter="\t")
    combined_data = pd.concat([combined_data, data], ignore_index=True)

output_file = "./features_a75_t90.tsv"
combined_data.to_csv(output_file, sep="\t", index=False)
