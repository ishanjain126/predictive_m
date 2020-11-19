import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import int64

# #Create a pandas DataFrame for the counts data set.
# df = pd.read_csv(r'C:\Users\hp\Desktop\bicycle.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
df = pd.read_csv(r'C:\Users\hp\Documents\predictive modelling\presentation\IRIS.csv')

# #Set up the X and y matrices
# y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
# y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
label_encoder = preprocessing.LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])
# print(df)
X = df.loc[:, df.columns != 'species' ]
y = df['species']
# y.reshape(-1, 1)
# X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#Using the statsmodels GLM class, train the Poisson regression model on the training data set.
poisson_training_results = sm.GLM(y_train.astype(float), X_train.astype(float), family=sm.families.Poisson()).fit()
# model = PoissonRegressor(alpha = 1e-12, max_iter = 300).fit(X_train, y_train)

# #Print the training summary.
print(poisson_training_results.summary())
# pred = model.predict(X_test)
# pred = int64(pred)
# acc = cross_val_score(model, X_train, y_train, cv = 5)
# print(acc.mean())

# #Make some predictions on the test data set.
poisson_predictions = poisson_training_results.get_prediction(X_test)
predictions_summary_frame = poisson_predictions.summary_frame()
# print(predictions_summary_frame)

predicted_counts=predictions_summary_frame['mean']
actual_counts = y_test

# Mlot the predicted counts versus the actual counts for the test data.
fig = plt.figure()
fig.suptitle("Actual v/s Predicted")
predicted = plt.plot(predicted_counts, 'go-', label='Predicted counts')
actual = plt.plot(actual_counts, 'ro-', label='Actual counts')
plt.legend()
plt.show()

#Show scatter plot of Actual versus Predicted counts
plt.clf()
fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts')
plt.scatter(x=predicted_counts, y=actual_counts, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')
plt.legend()
plt.show()

