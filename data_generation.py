import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

size_of_sample = 10

rows = np.random.choice(df.index.values, size_of_sample)

df_sample = df.loc[rows]

print(df_sample)