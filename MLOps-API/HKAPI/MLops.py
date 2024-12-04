# import library
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# import dataset
data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
features = df.columns

#Transpose
df.describe().T

#EDA
print(df.isna().sum()) # null handling
df.dropna(inplace=True)

print(df.duplicated().sum())
df.drop_duplicates(inplace=True) # duplicate handling

#Onehot encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# train_test split
y = df['charges']
X = df.drop(['charges'], axis=1)

# Drop NaN values from X and y before splitting
X.dropna(inplace=True)
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42,
                                                    )

# scale the features
mms = MinMaxScaler()
mms.fit(X_train) # learn min and max from train set for each column
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# Gradient Boosting Regression
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
print(r2_score(y_test, y_pred))

#save model
joblib.dump(gbr,"Insurance.pkl")
joblib.dump(features,"Insurance_features.pkl")