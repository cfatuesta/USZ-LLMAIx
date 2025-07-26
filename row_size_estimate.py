import csv
import sys
import re
import pandas as pd

df = pd.read_csv('master_abfrage_template.csv')
df["est_tokens"] = df.apply(lambda x: len(re.split(r'\s+', '\t'.join(map(str, x)))) * 3, axis=1)
# print(df[df.PATNR == "10499138"].est_tokens.sum())

print(df.groupby("PATNR")["est_tokens"].sum().sort_values(ascending=False).head(10))

print("\nEstimated tokens per row (5 percentile gaps), grouped by PATNR:")
# Show the distribution (5 percentile gaps) of estimated tokens per row
print(df.groupby("PATNR")["est_tokens"].sum().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
