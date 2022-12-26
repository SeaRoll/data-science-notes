## Chapter 1. Tabular data to pandas

```python
import pandas as pd

# import the data
landtemps = pd.read_csv('data.csv',                 # csv file
    skiprows=1,                                     # skip the first row
    parse_dates=[['month','year']])                 # parse specific columns into dates


landtemps.head(7)                                   # first 7 values
landtemps.dtypes                                    # data type of each column
landtemps.shape                                     # shape of dataframe
```

## Renaming columns & describing columns

```python
# fix the column name for the date
landtemps.rename(columns={'month_year':'measuredate'}, inplace=True)

# get description of column
landtemps['avgtemp'].describe()

# get sum of null values of each column
landtemps.isnull().sum()

# remove rows with missing values in specific column
landtemps.dropna(subset=['avgtemp'], inplace=True)
```

## Persisting data

```python
landtemps.to_csv('views/tempext.csv')
```
