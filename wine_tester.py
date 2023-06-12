import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("./sample_data/winequality-red.csv")

train_df, test_df = train_test_split(df, test_size=0.1)

X_train = train_df.drop('quality', axis=1)
y_train = train_df['quality']
X_test = test_df.drop('quality', axis=1)
y_test = test_df['quality']

clf = MLPClassifier()
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
print("Train MAE:", mean_absolute_error(y_train, train_predictions))
print("Test MAE:", mean_absolute_error(y_test, test_predictions))

new_data = np.array([10, 10, 0.13, 2.3, 0.076, 29.0, 40.0, 0.99574, 3.42, 0.75, 11.0])
new_data = new_data.reshape(1, -1)
prediction = clf.predict(new_data)
print("quality:", prediction[0])
