import pandas as pd

# Load the CSV
df = pd.read_csv("total_GA.csv")  # Replace with your actual file

# Column to check
column_name = 'Perfect_Trials'  # Replace with your column name

# Split the DataFrame
df_gt5 = df[df[column_name] > 5]     # Values > 5
df_le5 = df[df[column_name] <= 5]    # Values ≤ 5

# Optionally, save to separate files
df_gt5.to_csv("greater_than_5.csv", index=False)
df_le5.to_csv("less_equal_5.csv", index=False)
