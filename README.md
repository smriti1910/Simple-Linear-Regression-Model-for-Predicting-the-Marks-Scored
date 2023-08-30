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
![OUTPUT1](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/b43bc6fe-01ee-4f84-84c3-c7c003dbeac6)
![OUTPUT2](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/8d44e3bd-4ae0-4108-884d-30cd7af5dc75)
![OUTPUT3](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/7ce11faf-3151-41ac-be4b-f2e52cb1f886)
![OUTPUT4](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/81a1485a-4faf-4ccd-95d0-635a1cd618d9)
![OUTPUT5](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/c4e2db86-2acb-411d-92ae-2d43f0d815ba)
![OUTPUT6](https://github.com/smriti1910/Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/26f8ca2c-a9b7-48bc-af0d-e149bd020b5a)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
