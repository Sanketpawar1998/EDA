
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train_df = pd.read_excel("D:/Dataset/Flight Price Data_Train.xlsx")
train_df.head()

test_df = pd.read_excel("D:/Dataset/Flight Price Data Test_set.xlsx")
test_df.head()

final_df = train_df.append(test_df)
final_df.head()
final_df.tail()
final_df.info()

#Feature Engineering Process
final_df['Date']=final_df['Date_of_Journey'].str.split('/').str[0]
final_df['Month']=final_df['Date_of_Journey'].str.split('/').str[1]
final_df['Year']=final_df['Date_of_Journey'].str.split('/').str[2]
final_df.head(2)

final_df['Date'] = final_df['Date'].astype(int)
final_df['Month'] = final_df['Month'].astype(int)
final_df['Year'] = final_df['Year'].astype(int)
final_df.info()
final_df.drop('Date_of_Journey',axis=1,inplace=True)

final_df['Arrival_Time'].str.split(' ').str[0]
final_df['Arrival_Time'].apply(lambda x : x.split(' ')[0])
final_df['Arrival_Time'] = final_df['Arrival_Time'].apply(lambda x : x.split(' ')[0])

final_df.isnull().sum()
final_df['Arrival_hour'] = final_df['Arrival_Time'].str.split(':').str[0]
final_df['Arrival_min'] = final_df['Arrival_Time'].str.split(':').str[1]
final_df.head(1)

#Convert Datatype
final_df['Arrival_hour'] = final_df['Arrival_hour'].astype(int)
final_df['Arrival_min'] = final_df['Arrival_min'].astype(int)
final_df.info()
final_df.drop('Arrival_Time',axis=1,inplace=True)

final_df.head(1)

final_df['Dept_hour'] = final_df['Dep_Time'].str.split(':').str[0]
final_df['Dept_min'] = final_df['Dep_Time'].str.split(':').str[1]
final_df['Dept_hour'] = final_df['Dept_hour'].astype(int)
final_df['Dept_min'] = final_df['Dept_min'].astype(int)
final_df.drop('Dep_Time',axis=1,inplace=True)

final_df.info()
final_df['Total_Stops'].unique()
final_df['Total_Stops'].isnull().sum()

#Map Total Stop
final_df['Total_Stops']=final_df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stop':2,'3 stop':3,'4 stop':4,'nan':1})        
final_df.head()
final_df.drop('Route',axis=1,inplace=True)
final_df.head()

final_df['Additional_Info'].unique()

final_df['Duration_hour'] = final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]
final_df.head()
final_df.info()

final_df.drop(6474,axis=0,inplace=True)
final_df.drop(2660,axis=0,inplace=True)

final_df[final_df['Duration_hour']=='5m']

final_df['Duration_hour'] = final_df['Duration_hour'].astype('int')
final_df['Duration_hour']*60
final_df.info()

final_df.drop('Duration',axis=1,inplace=True)
final_df.head()

final_df['Airline'].unique()

#Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

final_df['Airline']=labelencoder.fit_transform(final_df['Airline'])
final_df['Source']=labelencoder.fit_transform(final_df['Source'])
final_df['Destination']=labelencoder.fit_transform(final_df['Destination'])
final_df['Additional_Info']=labelencoder.fit_transform(final_df['Additional_Info'])

final_df.shape

final_df.head(2)

from sklearn.preprocessing import OneHotEncoder
onehotencoding=OneHotEncoder()

final_df['Airline'].get_dummies()
pd.get_dummies(final_df,columns=["Airline","Source","Destination","Additional_Info"] ,drop_first = True)

