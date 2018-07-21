************************************
AUTHOR: CHIRANJEEVI TAPAL
************************************
EMAIL: CHIRANJEEVI.TAPAL@OUTLOOK.COM
************************************
# Real-Time-Data-Predictions
Real Time Data Predictions: A machine learning model

This is a simple machine learning model I made which predicts the prices of real estate based on certain parametres such as number of bedrooms, plot size and other key factors affecting the prices of houses.
This can be further enhanced to enhance any other metrics.
It makes use of the random forest model to predict prices. I have included the datasets which are needed to train the model and validate it later.

File description:

1. submission_file.csv - my final predictions file

2. train.csv - dataset to train or fit the model

3. test.csv - dataset to test the model and predict the prices

4. trainer.py - the python program file that trains with the training data. Result model is persisted using joblib pickle. 

5. tester.py - the python program file that tests the program. It loads the joblib pickle that is created by trainier for testing.

Data sources - www.kaggle.com
