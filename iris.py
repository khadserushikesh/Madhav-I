import pandas as pd 
import numpy  as np   
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split   
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score  
import warnings
warnings.filterwarnings('ignore')   
print('libraries imported !')

i=load_iris() 
x=i.data 
y=i.target 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=47)  
from sklearn.svm import SVC   
svc=SVC() 
svc.fit(x_train,y_train) 
print('accuracy score : ',accuracy_score(y_test,svc.predict(x_test)))   


import joblib as j   
j.dump(svc,'svc.pkl')