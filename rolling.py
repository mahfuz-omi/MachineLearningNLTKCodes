import pandas_test as pd
import matplotlib.pyplot as plt

data = [3.0, 20.0, 50.0, 104.0,172.0,252.0,350.0]
s1 = pd.Series(data)
df1 = pd.DataFrame(data)
#print(df1)
print(s1)
s1.rolling(window=2).sum
print(s1)
# print(s1)