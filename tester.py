import pandas as pd  # data processing, CSV file I/O
from sklearn.externals import joblib

forest_model = joblib.load('data/persistence.pkl')
test = pd.read_csv('data/test1.csv')  # Read the test data
predict_cols = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predict_cols]
predict_vals = forest_model.predict(test_X)  # make predictions using the model and test values
print("Making predictions for the following houses:")

print(test_X)
print("The predictions are")
print(predict_vals)  # print predicted values
