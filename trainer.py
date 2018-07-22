import pandas as pd  # data processing, CSV file I/O
from sklearn.ensemble import RandomForestRegressor  # library for Random Forest model
from sklearn.externals import joblib

predict_cols = ['Complexity', 'Time_Difference', 'Designation', 'Positive_Feedback', 'Negative_Feedback']

train_file_path = 'data/employee_train_data.csv'  # store file path
train = pd.read_csv(train_file_path)  # read the training data

y = train.Performance_Rating  # set single column salePrice as target column

train_X = train[predict_cols]  # set list of columns predic_cols as predictors
# train_X = pd.get_dummies(train_X)
pd.get_dummies(train_X['Designation'], prefix=['Designation'])
train_X = pd.concat([train_X, pd.get_dummies(train_X['Designation'], prefix='Designation')], axis=1)
train_X.drop(['Designation'], axis=1, inplace=True)

print(train_X.columns)

forest_model = RandomForestRegressor()  # create random forest model
forest_model.fit(train_X, y)  # train the model using predictors and target values

joblib.dump(forest_model, 'data/persistence_employee.pkl')
