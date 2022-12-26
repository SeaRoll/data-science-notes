# Data cleaning for ML
Data cleaning is a key part of data science, but it can be deeply frustrating. 
Why are some of your text fields garbled? What should you do about those missing values? 
Why aren’t your dates formatted correctly? How can you quickly clean up inconsistent data entry?

## Handling missing values

```python
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()
```

```python
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)
```

*remember, Is this value missing because it wasn't recorded or because it doesn't exist?*

```python
# remove all the rows that contain a missing value
nfl_data.dropna()

# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)

# replace all NA's with 0
subset_nfl_data.fillna(0)
```

