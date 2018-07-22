import pandas as pd  # data processing, CSV file I/O
from sklearn.ensemble import RandomForestRegressor  # library for Random Forest model
from sklearn.externals import joblib

predict_cols = ['SUBDIVISION', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV',
                'DEC']

train_file_path = 'data/rainfall_in_india_1901-2015_train_data.csv'  # store file path
train = pd.read_csv(train_file_path)  # read the training data

y = train.ANNUAL  # set single column salePrice as target column

train_X = train[predict_cols]  # set list of columns predic_cols as predictors
# train_X = pd.get_dummies(train_X)
pd.get_dummies(train_X['SUBDIVISION'], prefix=['SUBDIVISION'])
train_X = pd.concat([train_X, pd.get_dummies(train_X['SUBDIVISION'], prefix='SUBDIVISION')], axis=1)
train_X.drop(['SUBDIVISION'], axis=1, inplace=True)

print(train_X.columns)

forest_model = RandomForestRegressor()  # create random forest model
forest_model.fit(train_X, y)  # train the model using predictors and target values

joblib.dump(forest_model, 'data/persistenceweather.pkl')
