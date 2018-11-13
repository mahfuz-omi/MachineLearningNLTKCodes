import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("datas/student_scores.csv")

print(df)

# plot(x,y,style)
plt.plot(df.index,df['Hours'].values,"o")
plt.title("point")
plt.xlabel("index")
plt.ylabel("Hours")
plt.show()

plt.scatter(df.index,df['Hours'].values)
plt.title("scatter")
plt.xlabel("index")
plt.ylabel("Hours")
plt.show()

print(np.corrcoef(df["Scores"].values,df['Hours'].values))


# df.corr() returns corelation matrix with all columns but no index
print(df.corr())

# [[1.         0.07622967]
#  [0.07622967 1.        ]]
#            Hours    Scores
# Hours   1.000000  0.976191
# Scores  0.976191  1.000000

print(list(df.columns.values))


plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()

