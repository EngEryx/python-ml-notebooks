#%%
import pandas as pd
import numpy as np
import matplotlib as plt

#%%
s = pd.Series([1,3,5,np.nan,6,8])
print(s)


#%%
dates =  pd.date_range('20190725', periods=6)
print(dates)


#%%
df = pd.DataFrame(np.random.randn(6,4), index=dates,columns=list('ABCD'))
print(df)

#%%
df2 = pd.DataFrame({
    'A':1.,
    'B':pd.Timestamp('20190725'),
    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
    'D':np.array([3] * 4, dtype='int32'),
    'E':pd.Categorical(['test','train','test','train']),
    'F':'foo'
})

print(df2)

print(df2.dtypes)

#%%

df.head()

#%%

df2.head()

#%%
df.tail(3)

#%%
df.index

#%%
df.describe()

#%%
df.T

#%%
df.sort_index(axis=1,ascending=False)

#%%
df.sort_values(by='B')

#%%
df['A']

#%%
# Slices the rows
df[0:3]

#%%
# Selection by label
df.loc[dates[0]]

#%%
# selecting on a multi axis by label
df.loc[:,['A','B']]

#%%
# Showing label slicing, including both endpoints
df.loc['20190725':'20190728',['A','B']]


#%%
#Getting a scalar value
df.loc[dates[0],'A']

#%%
#For getting a faster access to a scalar value
df.at[dates[0],'A']

#%%
# Selection by Position

# Select via position of the passed integers
df.iloc[3]

#%%
# For slicing rows explicitly
df.iloc[1:3,:]

#%%
#For slicing columns explicitly
df.iloc[:,1:3]


#%%
# Boolean indexing
df[df.A > 0]

#%%
# A where operation for getting data.
df[df > 0]



#%%
# Using the isin() method for filtering
df3 = df.copy()

#%%
df3['E'] = ['one','one','two','three','four','three']

#%%
df3

#%%
# Setting a new column automatically aligns the data by the indexes
s2 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20190725', periods=6))

s2

#%%
# setting value by label
df.at[dates[0],'A'] = 0

#%%
df

#%%
# Setting value by position
df.iat[0,1] = 0


#%%
df

#%%
df.loc[:,'D'] = np.array([5] * len(df))

#%%

df

#%%
#Where operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
df2

#%%

# Other useful functions
# Reindexing allows you to change/add/delete the index on a specified axis and returns a copy of the data
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1

#%%
# drop any rows that have missing data
df1.dropna(how='any')

#%%
df1.fillna(value=5)

#%%
pd.isnull(df1)

#%%
#Operations excludes missing data
# perform a descriptive statistic
df.mean()

#%%
df.mean(1)


#%%
# Dealing with objects with different dimensionality and need aligning.
# We can set pandas to automatically broadcast according to specified dimension.

s = pd.Series([1,2,3,np.nan,6,8], index=dates).shift(2)
s


#%%
df.sub(s,axis='index')

#%%
# Apply functions to data
df.apply(np.cumsum)

#%%
df.apply(lambda x: x.max() - x.min())


#%%
# Histogramming and Discretization
s = pd.Series(np.random.randint(0,7,size=10))
s

#%%
s.value_counts()

#%%
# String methods
s = pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])

s.str.lower()

#%%

# Merge

## Concat 
# Pandas provide various functionalities for easilycombinong together series, DataFrame and panel objects with various kinds of set logic for the indexes and relational algebra functionality in the case of join/merge-type operations.

df = pd.DataFrame(np.random.randn(10,4))
df

#%%
pieces = [df[:3],df[3:7],df[7:]]

#%%
pieces

#%%
pd.concat(pieces)

#%%
# Join
# SQL Style merges
left = pd.DataFrame({'key': ['foo','foo'], 'lval':[1,2]})
right =  pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})
left
#%%
right

#%%
pd.merge(left,right, on='key')

#%%
# Append rows to a  DataFrame
df = pd.DataFrame(np.random.randn(8,4), columns = ['A','B','C','D'])

df

#%%
s = df.iloc[3
]

#%%
df.append(s, ignore_index=True)

#%%
s

#%%
# Grouping
# Group by i.e splitting, applying, combining
df = pd.DataFrame({
    'A':['foo','bar','foo','bar',
         'foo','bar','foo','foo'],
    'B':['one','one','two','three','two','two','one','three'],
    'C':np.random.randn(8),
    'D':np.random.randn(8)    
})

df

#%%
# Grouping and then applying a function sum to the resulting groups
df.groupby('A').sum()

#%%
df.groupby(['A','B']).sum()


#%%
# Reshaping
tuples