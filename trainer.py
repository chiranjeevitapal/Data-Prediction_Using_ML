import pandas as pd  # data processing, CSV file I/O
from sklearn.ensemble import RandomForestRegressor  # library for Random Forest model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

predict_cols = ['Complexity', 'Time_Difference', 'Designation', 'Positive_Feedback', 'Negative_Feedback']
# read csv file
train_file_path = 'data/employee_train_data.csv'
df = pd.read_csv(train_file_path)  # read the training data

# set single column salePrice as target column
predict_df = df.Performance_Rating

# set list of columns predict_cols as predictors
train_df = df[predict_cols]

pd.get_dummies(train_df['Designation'], prefix=['Designation'])
train_df = pd.concat([train_df, pd.get_dummies(train_df['Designation'], prefix='Designation')], axis=1)
train_df.drop(['Designation'], axis=1, inplace=True)

print(train_df.columns)

# Create the X and y arrays
X = train_df.values
y = predict_df.values

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

forest_model = RandomForestRegressor()  # create random forest model
forest_model.fit(X_train, y_train)  # train the model using predictors and target values

joblib.dump(forest_model, 'data/persistence_employee.pkl')
