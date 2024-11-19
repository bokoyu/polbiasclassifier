import pandas as pd
import os

combined_folder = r"C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\combined_data"
output_file = "all_combined_dataset.csv"

combined_files = [f for f in os.listdir(combined_folder) if f.endswith('_combined.csv')]

all_data = pd.concat([pd.read_csv(os.path.join(combined_folder, f)) for f in combined_files], ignore_index=True)

all_data.to_csv(output_file, index=False)
print(f"All combined datasets saved to {output_file}")
