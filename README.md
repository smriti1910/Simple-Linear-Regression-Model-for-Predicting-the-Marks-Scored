# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages 
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SMRITI .B
RegisterNumber:  212221040156
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
dataset.head()
dataset.tail()
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test, color="blue");

plt.plot(x_test, reg.predict(x_test), color="silver")

plt.title("Test set (H vs 5)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print('RMSE=',rmse)
b=np.array([[10]])
y_pred1=reg.predict(b)
print(y_pred1)
```


## Output:
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/bb506528-fdb5-47c5-a5d1-3d2500b7cbb4)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/ea7ea8d1-68c2-4ecb-954d-a727a65f88f0)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/b0bd60c7-9c5b-437a-a1b1-b75f9415517e)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/eacd8e42-0bda-441f-a2da-0e6f5e3e4eb5)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/7d0dd019-93c1-481d-a62a-22c71e280ada)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/b0905040-623d-41b6-a526-1a52ff9f0a25)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/dd21556f-6666-449f-8a6b-5ea7eae89e41)
![image](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/35b96fde-da13-46dd-ba28-ccc25437eeea)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
