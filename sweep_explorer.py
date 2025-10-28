import os
import glob
import numpy as np
import pandas as pd

# Base directory where all sweep results are stored
base_dir = './output_dir/datasets/datasets_4526628/hyper_sweep'

all_rows = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.npz'):
            file_path = os.path.join(root, file)
            # Extract sweep values from folder names (assuming 'param_xxx' structure)
            params = {}
            for part in root[len(base_dir):].split(os.sep):
                if '_' in part:
                    key, val = part.split('_', 1)
                    params[key] = val
            # Load MI results and labels
            try:
                npz = np.load(file_path)
                mi_labels = npz['MI_labels']
                mi_results = npz['MI_results']
                # If saved as bytes, decode
                mi_labels = [str(l, 'utf-8') if isinstance(l, bytes) else str(l) for l in mi_labels]
                # Make a dict of MI_label: MI_value
                mi_dict = dict(zip(mi_labels, mi_results))
                # Combine params and MI
                row = {**params, **mi_dict}
                all_rows.append(row)
            except Exception as e:
                print(f"Failed on {file_path}: {e}")

# Build DataFrame
df = pd.DataFrame(all_rows)

# Optional: Save to CSV for analysis
csv_path = os.path.join(base_dir, "MI_sweep_results.csv")
df.to_csv(csv_path, index=False)
print("Aggregated table saved to MI_sweep_results.csv")
print(df.head())
