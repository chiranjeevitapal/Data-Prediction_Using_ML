import pandas as pd  # data processing, CSV file I/O
from sklearn.externals import joblib

gb_model = joblib.load('data/persistence_employee.pkl')

# Read Input actual test input file
df_actual = pd.read_csv('data/employee_test_data.csv')
# Read sample excel will all designations. This is to let pandas run the test.
df_dummy = pd.read_csv('data/employee_all_designatins_test_data.csv')

# Append sample to input file
df_actual = df_actual.append(df_dummy)

# Fill empty cells
df_actual['Positive_Feedback'].fillna(False, inplace=True)
df_actual['Negative_Feedback'].fillna(False, inplace=True)
df_actual.fillna(0, inplace=True)

predict_cols = ['Complexity', 'Time_Difference', 'Designation', 'Positive_Feedback', 'Negative_Feedback']

# Treat the test data in the same way as training data. In this case, pull same columns.
df_predict = df_actual[predict_cols]

# On hot encode Categorical data. In this case Designation is Categorical.
pd.get_dummies(df_predict['Designation'], prefix=['Designation'])
df_predict = pd.concat([df_predict, pd.get_dummies(df_predict['Designation'], prefix='Designation')], axis=1)
df_predict.drop(['Designation'], axis=1, inplace=True)

# Get prediction for employee ratings from the trained pickle file
predicted_ratings = gb_model.predict(df_predict)

# Add a new column in the df_predict frame for prediction ratings
df_predict['Employee_Rating'] = predicted_ratings

# Merge actual data frame and predicted data frame to get full data for each employee
# df_merge = pd.merge(df_actual, df_predict[['PREDS']], how='inner', left_index=True, right_index=True)

df_merge = pd.merge(df_actual.assign(key=df_actual.groupby(level=0).cumcount()).reset_index(),
                    df_predict.assign(key=df_predict.groupby(level=0).cumcount()).reset_index(),
                    how='inner', on=['index', 'key']). \
    drop('key', 1).set_index('index')

print('Data after merge')
print(df_merge)

# Remove all the rows that have dummy assignees/employees designations that were added to make pandas happy
print('Remove dummy records')
df_without_fake_assignees = df_merge[df_merge.Assignee.str.contains("dummy") == False]
print(df_without_fake_assignees)

# Calculate mean performance of each assignee/employee and assign it to a new data frame
mean_performance_df = df_without_fake_assignees.groupby('Assignee').mean()
print(mean_performance_df)  # This has e final output

# Save the final employee performance predictions data to a csv
mean_performance_df.to_csv('data/employee_performance_prediction_output.csv')
