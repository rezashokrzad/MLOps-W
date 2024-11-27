import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# load data
data = fetch_california_housing()
X, y = data.data, data.target
features = data.feature_names

#X = pd.DataFrame(X, columns=features)
#print(X.head(2))

#label = pd.DataFrame(y, columns=["price"])
#print(label.head(2))

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42
    )

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

sample = np.array([[8.2, 41, 6.9, 1.02, 322, 2.5, 37.88, -122.23]])
print(round(lr.predict(sample)[0], 2))

joblib.dump(lr, "california_model.pkl")
joblib.dump(features, "california_features.pkl")


