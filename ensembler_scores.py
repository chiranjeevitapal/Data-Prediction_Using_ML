import pandas as pd  # data processing, CSV file I/O
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

predict_cols = ['Complexity', 'Time_Difference', 'Designation', 'Positive_Feedback', 'Negative_Feedback']
# read csv file
train_file_path = 'data/employee_train_data.csv'
df = pd.read_csv(train_file_path)  # read the training data

# Fill empty cells
df['Positive_Feedback'].fillna(False, inplace=True)
df['Negative_Feedback'].fillna(False, inplace=True)
df.fillna(0, inplace=True)

# set single column salePrice as target column
predict_df = df.Performance_Rating

# set list of columns predict_cols as predictors
train_df = df[predict_cols]

pd.get_dummies(train_df['Designation'], prefix=['Designation'])
train_df = pd.concat([train_df, pd.get_dummies(train_df['Designation'], prefix='Designation')], axis=1)
train_df.drop(['Designation'], axis=1, inplace=True)

# Create the X and y arrays. X has input columns to predict. y has only the column that needs to be learnt
X = train_df.values
y = predict_df.values

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# forest_model = RandomForestRegressor()  # create random forest model

rf = RandomForestRegressor()
gb = GradientBoostingRegressor(loss='huber', max_depth=6)
ada_tree_backing = DecisionTreeRegressor()
ab = AdaBoostRegressor(ada_tree_backing, learning_rate=1, loss='exponential', n_estimators=3000)

rf_score = cross_val_score(rf, X_train, y_train)
gb_score = cross_val_score(gb, X_train, y_train)
ab_score = cross_val_score(ab, X_train, y_train)

# precision = cross_val_score([rf], X, y, cv=10)

print("RF Precision: " + str(round(100 * rf_score.mean(), 2)) + "%")
print("GB Precision: " + str(round(100 * gb_score.mean(), 2)) + "%")
print("AB Precision: " + str(round(100 * ab_score.mean(), 2)) + "%")

# ab.fit(X_train, y_train)  # train the model using predictors and target values

# Dump the model to a pickle file
# joblib.dump(ab, 'data/persistence_employee.pkl')
