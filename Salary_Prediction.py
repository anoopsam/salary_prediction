import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.api import OLS

df = pd.read_csv (r'C:\Users\anoop.samdani\Downloads\Teacher_salary_csv.csv')
# Create dummy variables for 2 columns
df_PaymentMethod = pd.get_dummies(df['Campus'])
df_Churn = pd.get_dummies(df['Highest Degree'])
#Concat new columns to original dataframe 
df_concat = pd.concat([df, df_PaymentMethod, df_Churn], axis=1)
df_concat.drop(['Campus', 'Goa','Highest Degree' , 'Graduation'], inplace=True, axis=1)
x=df_concat.iloc[:,[6,7,8,9,10,12,13,14,15]]
y=df_concat.iloc[:,11]

regressor = LinearRegression()
regressor.fit(x,y)
y_pred= regressor.predict(x)
print(y)
print(y_pred)
results = sm.OLS(y,x).fit()
print (results.summary()) 
