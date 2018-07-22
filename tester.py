import pandas as pd  # data processing, CSV file I/O
from sklearn.externals import joblib

forest_model = joblib.load('data/persistence_employee.pkl')

# Read Input file
df_actual = pd.read_csv('data/employee_test_data.csv')
# Read sample excel will all designations
df_dummy = pd.read_csv('data/employee_all_designatins_test_data.csv')

# Append sample to input file
df_actual = df_actual.append(df_dummy)
print('data after append')
print(df_actual)

# Fill empty cells
df_actual['Positive_Feedback'].fillna(False, inplace=True)
df_actual['Negative_Feedback'].fillna(False, inplace=True)
df_actual.fillna(0, inplace=True)

predict_cols = ['Complexity', 'Time_Difference', 'Designation', 'Positive_Feedback', 'Negative_Feedback']

# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = df_actual[predict_cols]

pd.get_dummies(test_X['Designation'], prefix=['Designation'])
test_X = pd.concat([test_X, pd.get_dummies(test_X['Designation'], prefix='Designation')], axis=1)
test_X.drop(['Designation'], axis=1, inplace=True)

# Get prediction values
predict_vals = forest_model.predict(test_X)  # make predictions using the model and test values

print('Predict values')
print(predict_vals)

# Add a new column in the df for prediction values
test_X['PREDS'] = predict_vals

# Create a new df with the merged columns
# df_merge = pd.merge(df_actual, test_X[['PREDS']], how='inner', left_index=True, right_index=True)

df_merge = pd.merge(df_actual.assign(key=df_actual.groupby(level=0).cumcount()).reset_index(),
                    test_X.assign(key=test_X.groupby(level=0).cumcount()).reset_index(),
                    how='inner', on=['index', 'key']). \
    drop('key', 1).set_index('index')

print('Data after merge')
print(df_merge)

print('Data after predictions')
df_out = df_merge[df_merge.Assignee.str.contains("dummy") == False]
print(df_out)

# Calculate average performance of an employee
merge_df_average_by_state = df_out.groupby('Assignee').mean()
print(merge_df_average_by_state)

# Save the predictions data to a csv
merge_df_average_by_state.to_csv('data/prediction_output.csv')
