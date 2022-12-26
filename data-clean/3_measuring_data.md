## What this file covers:

The recipes in this chapter offer techniques for determining if the data 
is in good enough shape to begin the analysis, so that even if we cannot say, 
"it looks fine," we can at least say, 
"I'm pretty sure I have identified the main issues, and here they are."

#### Checklist when getting new data
- How are the rows of the dataset uniquely identified? (What is the unit of analysis?)
- How many rows and columns are in the dataset?
- What are the key categorical variables and the frequencies of each value?
- How are important continuous variables distributed?
- How might variables be related to each other – for example, how might the distribution of continuous variables vary according to categories in the data?
- What variable values are out of expected ranges, and how are missing values distributed?

## Taking a first look

Often, there is a unique identifier when working with individuals as the unit of analysis. 
This is a good candidate for an index. It makes selecting a row by that identifier easier. 
Rather than using the statement `nls97.loc[personid==1000061]` to get the row for that 
person, we can use `nls97.loc[1000061]`.

```python
import pandas as pd
import numpy as np

nls97 = pd.read_csv("data/nls97.csv")

nls97.set_index("personid", inplace=True) # set id as index
nls97.index                               # list of index
nls97.shape                               # shape of dataframe
nls97.index.nunique()                     # number of unique index (should be the same as rows!)
nls97.info()                              # get information of each column (non-null count, dtype, etc)
nls97.head(2).T                           # show first two rows as a list (instead of column)
nls97.sample(2, random_state=1).T         # can also do this to get random sample in the table
```
## Selecting and organizing columns

We want to Focus on relevant variables and group or limit columns according to their 
relationships or importance when cleaning or analyzing data to avoid confusion.

```python
# swap all object datatypes to category.
# read more on https://stackoverflow.com/questions/30601830/when-to-use-category-rather-than-object
# on why it is used.
nls97.loc[:, nls97.dtypes == 'object'] = \
  nls97.select_dtypes(['object']). \
  apply(lambda x: x.astype('category'))
```

#### Selecting single column

```python
analysisdemo = nls97['gender']    # returns a series
analysisdemo = nls97[['gender']]  # returns a dataframe
```

#### Selecting multiple columns
```python
analysisdemo = nls97[['gender', 'maritalstatus', 'highestgradecompleted']]  # select three columns (returns dataframe)
```

#### Selecting column that are like a name
```python
analysiswork = nls97.filter(like="weeksworked") # returns a dataframe of all columns with name like that
```

#### Selecting columns based on data
```python
# select based on dtype category
analysiscats = nls97.select_dtypes(include=["category"])
analysiscats.info()

# select based on dtype number
analysisnums = nls97.select_dtypes(include=["number"])
analysisnums.info()
```

#### Merge list of keys after each other (like sorting columns)
```python
nls97 = nls97[demoadult + demo + highschoolrecord + govresp + weeksworked + colenr]
```

## Selecting rows
```python
# use slicing to select a few rows
nls97[1000:1004].T
nls97[1000:1004:2].T

# select a few rows using loc (by id) and iloc (by index)
nls97.loc[[195884,195891,195970]].T
nls97.loc[195884:195970].T
nls97.iloc[[0]].T
nls97.iloc[[0,1,2]].T

nls97.nightlyhrssleep.quantile(0.05)    # how many hrs lowest 5% slept
nls97.nightlyhrssleep.count()           # out of how many

# select rows where column was less or equal than 4
lowsleep = nls97.loc[nls97.nightlyhrssleep<=4]

# multiple conditions
lowsleep3pluschildren = nls97.loc[(nls97.nightlyhrssleep<=4) & (nls97.childathome>=3)]

# select rows based on multiple conditions and also select columns
lowsleep3pluschildren = nls97.loc[(nls97.nightlyhrssleep<=4) & (nls97.childathome>=3), ['nightlyhrssleep','childathome']]
```

## Generating frequencies for categorical variables
Frequency distributions (crosstabs) can help you understand and explore a DataFrame. 
The more you do, the better you will understand the data.

```python
# show the names of columns with category data type and check for number of missings
catcols = nls97.select_dtypes(include=["category"]).columns
nls97[catcols].isnull().sum()

# show counts for each category value
nls97.maritalstatus.value_counts()
nls97.maritalstatus.value_counts(sort=False)
nls97.maritalstatus.value_counts(sort=False, normalize=True) # percentage

# do percentages for all government responsibility variables
nls97.filter(like="gov").apply(pd.value_counts, normalize=True)

# do percentages for all government responsibility variables for people who are married
nls97[nls97.maritalstatus=="Married"].filter(like="gov").apply(pd.value_counts, normalize=True)
```

#### Saving frequencies of all categorical values
```python
# do frequencies and percentages for all category variables in data frame
freqout = open('views/frequencies.txt', 'w') 
for col in nls97.select_dtypes(include=["category"]):
  print(col, "----------------------", "frequencies",
  nls97[col].value_counts(sort=False),"percentages",
  nls97[col].value_counts(normalize=True, sort=False),
  sep="\n\n", end="\n\n\n", file=freqout)

freqout.close()
```

## Generating summary statistics for continuous variables

