import pandas as pd

file_path = "./Data/SBIN.csv"
df = pd.read_csv(file_path, skiprows=1)  # Skip first header row
print("CSV Columns:", df.columns)
