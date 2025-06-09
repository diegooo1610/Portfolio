import pandas as pd
import sqlite3
import os

# Path to your CSV file
csv_path = "Py_Notebooks/cleaned_5yr_stock_data_with_month.csv" # ✅ Update if needed

# Check if the CSV exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found: {csv_path}")

# Load CSV into DataFrame
df = pd.read_csv(csv_path)

# Create the SQLite database inside the SQLite/ folder
db_path = "SQLite/stock_data.db"
conn = sqlite3.connect(db_path)

# Write the DataFrame to the 'stocks' table
df.to_sql("stocks", conn, if_exists="replace", index=False)

conn.close()

print(f"✅ Data loaded into 'stocks' table in '{db_path}'")
