from sklearn.externals import joblib
import pandas as pd

# Load the model we trained previously
model = joblib.load('trained_emp_performance_model.pkl')

# Read Input file
df = pd.read_csv('Emp_Data_33.csv')

# Read sample excel will all designations
df2 = pd.read_csv('All_Designations_Sample.csv')

# Append sample to input file
df = df.append(df2)
print(df)

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['Designation'])

# Save the input file after one hot encoding
features_df.to_csv('Emp_33_Sorted.csv', index=False)

# Delete last 5 rows of sample data after generating input file with all required prediction columns
df = df[:-5]
print(df)

# Delete Assignee column from input file
del features_df['Assignee']

# Save the input file after one hot encoding
features_df.to_csv('Emp_44_Sorted.csv', index=False)

# Run the model and make a prediction for each case of employee
predicted_emp_perf_values = model.predict(features_df)

# Convert the predicted data from NDArray to Dataframe
predicted_emp_perf_values = pd.DataFrame(predicted_emp_perf_values)

# Create a new dataframe merging the Assignee column from input and the predicted value column from output
merge_df = pd.merge(df[['Assignee']],predicted_emp_perf_values,how='left', left_index=True, right_index=True)
print(merge_df)
# Calculate average performance of an employee
merge_df_average_by_assignee = merge_df.groupby('Assignee').mean()
print (merge_df_average_by_assignee)





# Predicting the performance of employee for each case
# for i in range(len(predicted_emp_perf_values)):
#     predicted_value = predicted_emp_perf_values[i]
#     print("Employee has an estimated performance rating of ${:,.2f}".format(predicted_value))

# print(predicted_emp_perf_values)
# print (df[['Assignee']])
# merge_df = df[['Assignee']].append(predicted_emp_perf_values)
# print(merge_df.sort_values(['Assignee']))
# print(predicted_emp_perf_values.mean())
# print(predicted_emp_perf_values.std())
# for row in merge_df.iterrows():
# print("Employee has an estimated pmerge_df.iteritems()erformance rating of ${:,.2f} - %s".format(predicted_value), df['Designation' : i])
# merge_df = pd.concat([df['Assignee'], predicted_emp_perf_values], axis=1, ignore_index=True)
# Sort DataFrame
# df = df.sort_values(['Assignee','Complexity'])
# print(df)