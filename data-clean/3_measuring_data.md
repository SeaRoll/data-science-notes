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



