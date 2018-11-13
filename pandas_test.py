import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df)

#print(df.values)


#print(df.values[0])

# df = df.values[0:4]
# print(df)

# df = df.values[0:3,0:2]
# print(df)


#print(df.values[:,1][0])

#print(df.iloc[[1],[0,1,2]])
#print(df.iloc[0])

enumerate_data = enumerate([1,3,5,7])

for item in enumerate_data:
    print(item)

print(list(enumerate_data))


