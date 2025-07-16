import pandas as pd
import glob
import os

# Set your folder path
folder_path = '/home/ananya/Desktop/PID TUNING/PID-tuning/GA CSV/'  # 🔁 Change this
csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

# Read and concatenate all files as-is (including repeated headers)
dfs = [pd.read_csv(f, header=0, on_bad_lines='skip') for f in csv_files]

# Combine
combined_df = pd.concat(dfs, ignore_index=True)

# Save to new CSV
combined_df.to_csv('combined_with_headers.csv', index=False)

print(f"✅ Combined {len(csv_files)} files. Total rows (including any repeated headers): {len(combined_df)}")
