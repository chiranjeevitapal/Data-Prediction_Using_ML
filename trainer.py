import pandas as pd  # data processing, CSV file I/O
from sklearn.ensemble import RandomForestRegressor  # library for Random Forest model
from sklearn.externals import joblib

train_file_path = 'data/train.csv'  # store file path
train = pd.read_csv(train_file_path)  # read the training data
print(train.describe())  # print data
print(train.columns)  # print name of columns

y = train.SalePrice  # set single column salePrice as target column
predict_cols = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X = train[predict_cols]  # set list of columns predic_cols as predictors

forest_model = RandomForestRegressor()  # create random forest model
forest_model.fit(train_X, y)  # train the model using predictors and target values

joblib.dump(forest_model, 'data/persistence.pkl')
