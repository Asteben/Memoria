import pandas as pd
df = pd.read_json (r'dataset/dataset.jsonl')
df.to_csv (r'dataset/datasetFinal.csv', index = None)