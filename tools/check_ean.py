import pandas as pd
from gtin import has_valid_check_digit


df = pd.read_csv("processed.csv")

valid = []
for index, row in df.iterrows():
    if has_valid_check_digit(row['EAN_RAW']):
    	valid.append("VALID")
    else:
    	valid.append("ERROR")

df.insert(13, "CHECK", valid, True)
df.to_csv("checked.csv")