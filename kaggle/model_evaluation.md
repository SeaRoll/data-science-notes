# Permutation Importance
One of the most basic questions we might ask of a model is: What features have the biggest impact on predictions?
This concept is called feature importance.There are multiple ways to measure feature importance. Some approaches 
answer subtly different versions of the question above. Other approaches have documented shortcomings.

#### Process
With this insight, the process is as follows:

1. Get a trained model.
2. Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
3. Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.

#### Using ELI5 Library to show weights
```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```
