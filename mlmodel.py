

Loan Prediction

#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

encoder = preprocessing.LabelEncoder()

#importing dataset
data = pd.read_csv('/content/datasets_of Loan Prediction.csv')

data

data.info()

Data Cleaning

#importing dataset

data = data.fillna(np.nan,axis=0)

#encoding catagorical values

data['Gender'] = encoder.fit_transform(data['Gender'].astype(str))
data['Married'] = encoder.fit_transform(data['Married'].astype(str))
data['Dependents'] = encoder.fit_transform(data['Dependents'].astype(str))
data[['Education']] = encoder.fit_transform(data['Education'].astype(str))
data[['Self_Employed']] = encoder.fit_transform(data['Self_Employed'].astype(str))
data[['Property_Area']] = encoder.fit_transform(data['Property_Area'].astype(str))
data[['Credit_History']] = encoder.fit_transform(data['Credit_History'].astype(str))
data[['Loan_Amount_Term']] = encoder.fit_transform(data['Loan_Amount_Term'].astype(str))
data[['LoanAmount']] = encoder.fit_transform(data['LoanAmount'].astype(str))
data[['Loan_Status']] = encoder.fit_transform(data['Loan_Status'].astype(str))


from sklearn.preprocessing import LabelEncoder
category= ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'] 

labelencoder_X= LabelEncoder()
for i in category:   
    data[i] = labelencoder_X.fit_transform(data[i]) 
    data.dtypes


print(data.dtypes)

#printing first few entry
print(data.head())

#finding out null values
print(data.isna().sum())

#importing libraries
from sklearn.metrics import confusion_matrix as cm

#Extracting the data set into X and Y values 
X = data[['Gender','Married','Dependents','Education','Self_Employed','Property_Area' ]]
Y = data.iloc[:,-1]
print(X)
print(Y)

#Spliting data set into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,Y_train)

#predicting the values
pred = np.array(classifier.predict(X_test))
# Predicting the Test set results
y_pred = classifier.predict(X_test)



y_pred


ma = classifier.score(X_test,Y_test)

print('Accuracy: ',ma)

# we can print the accuracy of the classification problem and create the confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(Y_test,pred)
print(cm)

# Now we will use the mlxtend library to plot the confusion matrix
# and to visualize the output
from mlxtend.plotting import plot_confusion_matrix,plot_decision_regions
plot_confusion_matrix(cm)
plt.show()


Web App Using Flask

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
import pickle
pickle.dump(classifier, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,0,0,0,1,1]]))

from flask import Flask, render_template,request
