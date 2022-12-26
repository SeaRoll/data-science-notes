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

# Partial Dependence Plots
While feature importance shows what variables most affect predictions, 
partial dependence plots show how a feature affects predictions.
This is useful to answer questions like:

- Controlling for all other house features, what impact do longitude and latitude have on home prices? To restate this, how would similarly sized houses be priced in different areas?
- Are predicted health differences between two groups due to differences in their diets, or due to some other factor?

#### Process
We will use the fitted model to predict our outcome (probability their player won "man of the match"). But we repeatedly alter the value for one variable to make a series of predictions. We could predict the outcome if the team had the ball only 40% of the time. We then predict with them having the ball 50% of the time. Then predict again for 60%. And so on. We trace out predicted outcomes (on the vertical axis) as we move from small values of ball possession to large values (on the horizontal axis).

#### Using PDPBox library
```python
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
```
![Screen Shot 2022-12-26 at 21 11 00](https://user-images.githubusercontent.com/26680151/209580637-f2ff19ab-8ffb-4d85-8a37-06f8a7a9af50.png)

The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.
A blue shaded area indicates level of confidence

# SHAP Values
SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature.
Example: A model says a bank shouldn't loan someone money, and the bank is legally required to explain the basis for each loan rejection.

#### Usage

```python
import shap  # package used to calculate Shap values

row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
my_model.predict_proba(data_for_prediction_array)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# Plot
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
```

![Screen Shot 2022-12-26 at 21 15 25](https://user-images.githubusercontent.com/26680151/209580863-7f20f263-a3db-4b63-b974-0965803b1ce5.png)
