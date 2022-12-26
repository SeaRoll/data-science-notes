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

### From intermediate machine learning

1) A Simple Option: Drop Columns with Missing Values
The simplest option is to drop columns with missing values.

2) A Better Option: Imputation
Imputation fills in the missing values with some number. For instance, we can fill in the mean value along each column.
The imputed value won't be exactly right in most cases, but it usually leads to more accurate models 
than you would get from dropping the column entirely.

3) An Extension To Imputation
Imputation is a common method for handling missing values, 
but imputed values may not always be accurate and missing 
values may indicate unique characteristics in the data. In 
these cases, considering which values were originally 
missing may improve model performance.

#### Approach 1 (Drop Columns with Missing Values)

```python
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
```

#### Approach 2 (Imputation)

```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_df = pd.DataFrame(my_imputer.fit_transform(df))

# Imputation removed column names; put them back
imputed_df.columns = df.columns
```

#### Approach 3 (Extension to imputation)

```python

df_plus = df.copy()

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_df_plus = pd.DataFrame(my_imputer.fit_transform(df_plus))

# Imputation removed column names; put them back
imputed_df_plus.columns = df_plus.columns
```

## Scaling and normalization

```python
# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)
```

#### Scaling
This means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1

For example, you might be looking at the prices of some products in both Yen and US Dollars. 
One US Dollar is worth about 100 Yen, but if you don't scale your prices, 
methods like SVM or KNN will consider a difference in price of 1 
Yen as important as a difference of 1 US Dollar!

```python
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0]) # columns to scale
```

#### Normalization
Scaling just changes the range of your data. Normalization is a more radical transformation. 
The point of normalization is to change your observations so that they can be described as a normal distribution.

```
Normal distribution: Also known as the "bell curve", this is a specific statistical distribution 
where a roughly equal observations fall above and below the mean, 
the mean and the median are the same, and there are more observations 
closer to the mean. The normal distribution is also known as the Gaussian distribution.
```

Normalize data for machine learning or statistical techniques that 
assume normality, such as LDA and Gaussian naive Bayes. 
Techniques with "Gaussian" in the name often assume normality.

```python
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
```

![Screen Shot 2022-12-26 at 20 24 44](https://user-images.githubusercontent.com/26680151/209578220-04954fa8-c4c3-45bf-bc99-da5d5f610330.png)

## Parsing Dates

```python
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")

# use this if there are multiple date formats in a column
landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
```

















