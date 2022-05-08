import pandas as pd
import numpy as np

corona_GBy_data=pd.read_csv('Data/TestCorona-GBy60s.csv',names=['Unix','Quantity'])

print(corona_GBy_data.head(1)['Unix'].tolist()[0])