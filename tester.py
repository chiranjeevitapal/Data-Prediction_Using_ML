import pandas as pd  # data processing, CSV file I/O
from sklearn.externals import joblib
from sklearn import preprocessing

forest_model = joblib.load('data/persistenceweather.pkl')
test = pd.read_csv('data/rainfall_in_india_test_data.csv')  # Read the test data
predict_cols = ['SUBDIVISION', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV',
                'DEC']
# Create a label (category) encoder object
le = preprocessing.LabelEncoder()
test['SUBDIVISION'] = le.fit_transform(test['SUBDIVISION'])

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predict_cols]

predict_vals = forest_model.predict(test_X)  # make predictions using the model and test values

print(test_X)
print("The predictions are")
print(predict_vals)  # print predicted values
