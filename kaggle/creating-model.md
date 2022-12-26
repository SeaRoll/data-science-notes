# Creating models

#### Libraries to use
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
```

#### Import csv
```python
df = pd.read_csv('data.csv')
```

#### Split by features and target value
```python
features = [c for c in df.columns if c not in ['is_churned']]
X = df[features]
y = df['is_churned']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```

#### Train model

```python
# RandomForestRegressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)


# XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5)
my_model.fit(train_X, train_y,
         	eval_set=[(val_X, val_y)],
         	verbose=False)
```

#### Predict
```python
rf_predictions = rf_model.predict(val_X)
mae = mean_absolute_error(rf_predictions, val_y)
print(f"Mean Absolute Error: {mae * 100}%")
```
