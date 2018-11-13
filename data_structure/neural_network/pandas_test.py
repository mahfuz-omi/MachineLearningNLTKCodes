#https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
import pandas as pd

# return only the genres column
df = pd.read_csv("movie.csv",usecols=['genres','duration'])
pd.set_option('expand_frame_repr', False)


# return these column
#df = df[["director_name", "genres", "actor_1_name", "movie_title", "imdb_score", "plot_keywords"]]
print(df.head())

# data is mandatory...index is optional
series = pd.Series(data=[1,3,5,7],index=[1,2,3,4],name="omi")
print(series)
print('omi',series.iloc[1])
#
# print(type(series.values))
#
# print(series.values[1])

print(df.dtypes)

# apply for all column
print(df.describe())

# apply for only genres column
print(df['genres'].describe())


# selecting columns
#print(df['genres'])

#print(df.genres)

# have to use int
#print(df.iloc[:,0])

# have to use string
#print(df.loc[:,'genres'])

# select column and operations
# single selected column return series
#When a column is selected using any of these methodologies,
# a pandas.Series is the resulting datatype.
# A pandas series is a one-dimensional set of data.
# It’s useful to know the basic operations that can be carried out on these Series of data,
# including summing (.sum()), averaging (.mean()),
# counting (.count()), getting the median (.median()),
#  and replacing missing values (.fillna(new_value)).
print(df['duration'].sum())
print(df['duration'].count())

# looping call, called for each row of duration column, returns true or false for all row
print(df['duration'].isnull())

# number of null values in duration column
print(df['duration'].isnull().sum())

# cropping dataframe
list_columns = ['genres','duration']
print(df[list_columns])

print(df[['genres','duration']])

# selecting rows
print(df.iloc[0:11,:])

print(df.iloc[10,:])

# logical row selection
# select all rows where duration < 100
print(df[df["duration"] < 100])

# after selection all rows where duration < 100, more filters can be added
# here, first 5 rows with only the first column have been returned
print(df[df["duration"] < 100].iloc[0:5,0])


# Single selections using iloc and DataFrame
# Rows:
# data.iloc[0] # first row of data frame (Aleshia Tomkiewicz) - Note a Series data type output.
# data.iloc[1] # second row of data frame (Evan Zigomalas)
# data.iloc[-1] # last row of data frame (Mi Richan)
# # Columns:
# data.iloc[:,0] # first column of data frame (first_name)
# data.iloc[:,1] # second column of data frame (last_name)
# data.iloc[:,-1] # last column of data frame (id)

# Multiple row and column selections using iloc and DataFrame
# data.iloc[0:5] # first five rows of dataframe
# data.iloc[:, 0:2] # first two columns of data frame with all rows
# data.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
# data.iloc[0:5, 5:8] # first 5 rows and 5th, 6th, 7th columns of data frame (county -> phone1).

# Select rows with index values 'Andrade' and 'Veness', with all columns between 'city' and 'email'
# data.loc[['Andrade', 'Veness'], 'city':'email']
# # Select same rows, with just 'first_name', 'address' and 'city' columns
# data.loc['Andrade':'Veness', ['first_name', 'address', 'city']]
#
# # Change the index to be based on the 'id' column
# data.set_index('id', inplace=True)
# # select the row with 'id' = 487
# data.loc[487]

# Deleting columns
# Delete the "Area" column from the dataframe
# data = data.drop("Area", axis=1)
# # alternatively, delete columns using the columns parameter of drop
# data = data.drop(columns="area")
# # Delete the Area column from the dataframe in place
# # Note that the original 'data' object is changed when inplace=True
# data.drop("Area", axis=1, inplace=True).
# # Delete multiple columns from the dataframe
# data = data.drop(["Y2001", "Y2002", "Y2003"], axis=1)

# Delete the rows with labels 0,1,5
# data = data.drop([0,1,2], axis=0)
# # Delete the rows with label "Ireland"
# # For label-based deletion, set the index first on the dataframe:
# data = data.set_index("Area")
# data = data.drop("Ireland", axis=0). # Delete all rows with label "Ireland"
# # Delete the first five rows using iloc selector
# data = data.iloc[5:,]

# Rename columns using a dictionary to map values
# Rename the Area columnn to 'place_name'
# data = data.rename(columns={"Area": "place_name"})
# # Again, the inplace parameter will change the dataframe without assignment
# data.rename(columns={"Area": "place_name"}, inplace=True)
# # Rename multiple columns in one go with a larger dictionary
# data.rename(
#     columns={
#         "Area": "place_name",
#         "Y2001": "year_2001"
#     },
#     inplace=True
# )
# # Rename all columns using a function, e.g. convert all column names to lower case:
# data.rename(columns=str.lower)

# Output data to a CSV file
# Typically, I don't want row numbers in my output file, hence index=False.
# To avoid character issues, I typically use utf8 encoding for input/output.
#data.to_csv("output_filename.csv", index=False, encoding='utf8')

# sorting df
# sort by column value
# null values come last by default
print(df.sort_values('duration',ascending=False))

print(df.sort_values('duration',na_position='first'))

# sort by multiple column values
# ascending , null comes last by default
print(df.sort_values(['duration','genres']))

# iterate
# iterate in whole df
for index, row in df.iterrows():
    print (row["duration"],"omi", row["genres"])

# iterate in only a single column
for row in df['test_score']:
    if row > 95:
        print(row)

#If you wish to modify the rows you're iterating over, then df.apply is preferred:
# convert age to half age
def function_name(x):
    return x * 0.5

df['age_half'] = df.apply(lambda row: function_name(row['age']), axis=1,)


#https://www.oreilly.com/learning/handling-missing-data
#drop null values
#We cannot drop single values from a DataFrame;
# we can only drop full rows or full columns.
# Depending on the application, you might want one or the other,
# so dropna() gives a number of options for a DataFrame.

#By default, dropna() will drop all rows in which any null value is present:
# drop row where null
df.dropna()
#Alternatively, you can drop NA values along a different axis:
# axis=1 drops all columns containing a null value:

#drop column where null
df.dropna(axis=1)
#Keep in mind that to be a bit more clear,
# you can use axis='rows' rather than axis=0
# and axis='columns' rather than axis=1.

#We can fill NA entries with a single value, such as zero:


df.fillna(0)
#We can specify a forward-fill to propagate the previous value forward:


# forward-fill
df.fillna(method='ffill')
#Or we can specify a back-fill to propagate the next values backward:


# back-fill
df.fillna(method='bfill')

# drop the column (axis=1) where all values (all) are null
df.dropna(axis=1, how='all')

# drop the rows (axis=0) where all values (all) are null
df.dropna(axis=0, how='all')

# drop the column (axis=1) where any value (any) is null
df.dropna(axis=1, how='any')

# drop the rows (axis=0) where any value (any) is null
df.dropna(axis=0, how='any')

# any is by default

# iloc vs regular indexing(index column)

# return the value(not index) of the series where index = n(index column)
#series[n]

# return the value(not index) using integer indexing (overriding regular index column)= n-1 the data
#series.iloc[n]

# lowercase to all data in a series
# looping call
serius = series.str.lower()
df['column'] = df['column'].str.lower()

