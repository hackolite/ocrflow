import pandas as pd
raw = pd.read_csv("ocr_3.csv")
annotation = pd.read_csv("result.csv")
result = pd.merge(raw, annotation, how="left", on="Image")
result.to_csv("processed.csv")