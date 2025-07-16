import pandas as pd

# Load the CSV
df = pd.read_csv("ga5678.csv")  # Replace with your file name



# Drop a column
column_to_drop = 'Score'  # Replace with the column you want to drop
df.drop(columns=[column_to_drop], inplace=True)

# Save to new CSV
df.to_csv("ga5678.csv", index=False)
