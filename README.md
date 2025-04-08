# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection & Preprocessing
2.Select relevant features that impact placement 
3.Import the Logistic Regression model from sklearn. 
4.Train the model using the training dataset.
5.Use the trained model to predict placement for new student data.

## Program:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JEYAARIKARAN P
RegisterNumber:  212224240064
*/
```

## Output:

![image](https://github.com/user-attachments/assets/c99ce235-71b5-4f2c-a349-e75a6e91375e)


![image](https://github.com/user-attachments/assets/7e010984-20d1-48a2-9d20-bae1f2b8f15a)


![image](https://github.com/user-attachments/assets/7e0a3c12-edf2-43e6-b6ea-d07416b5ac20)


![image](https://github.com/user-attachments/assets/b73746fd-973f-4666-9e50-1efe6c7e022d)



![image](https://github.com/user-attachments/assets/aaab5b5f-0522-4d37-bba4-1ea93fabd10f)



![image](https://github.com/user-attachments/assets/177b30d9-8425-46ee-94d0-adbea4d35ff8)



![image](https://github.com/user-attachments/assets/c5f62335-9052-4e09-9950-936f248ca812)



![image](https://github.com/user-attachments/assets/64181729-d936-40bb-b5cb-6fe638258d9c)











## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
