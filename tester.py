import pandas as pd  # data processing, CSV file I/O
from sklearn.externals import joblib

forest_model = joblib.load('data/persistence_employee.pkl')
test = pd.read_csv('data/employee_all_designatins_test_data.csv')  # Read the test data
predict_cols = ['Complexity', 'Time_Difference', 'Designation', 'Positive_Feedback', 'Negative_Feedback']

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predict_cols]

pd.get_dummies(test_X['Designation'], prefix=['Designation'])
test_X = pd.concat([test_X, pd.get_dummies(test_X['Designation'], prefix='Designation')], axis=1)
test_X.drop(['Designation'], axis=1, inplace=True)

# Get prediction values
predict_vals = forest_model.predict(test_X)  # make predictions using the model and test values

# Add a new column in the df for prediction values
test_X['PREDS'] = predict_vals

# Create a new df with the merged columns
df_out = pd.merge(test, test_X[['PREDS']], how='left', left_index=True, right_index=True)

print('Data after predictions')
print(df_out)

# Calculate average performance of an employee
merge_df_average_by_state = df_out.groupby('Assignee').mean()
print(merge_df_average_by_state)
