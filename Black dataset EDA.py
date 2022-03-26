#Black Friday Dataset EDA And Feature Engineering

A retail company 'ABC Private Limited' wants to understand the customer purchase behaviour (specifically,purchase amount) againts various products of different categories . They have shared purchase summary of various customers for selected high volume products form last month.The data set also contains customers demographics(age,gender,marital status,city_type,stay_in_current_city),product details(product_id and product category) and total purchase_amount from last month
Now, they want build a model to predict the purchase amount of customer againts product which will help them to create personalized offer for customers againts different products

#Import Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

 #Importing Dataset
df_train = pd.read_csv("D:/Dataset/Black Friday Train.csv")
df_train.head()

#Import Test Data
df_test = pd.read_csv("D:/Dataset/Black Friday Test.csv")
df_test.head()

#Merge both train and test data
df1 = df_train.append(df_test)  #Adding train and test dataset
df1.head()

#Basic Code
df1.info()

df1.describe()

df1.drop(['User_ID'],axis=1,inplace=True)   #Drop user id from dataset
df1.head()

#Handling Categorical Feature for Gender

pd.get_dummies(df1['Gender'])
df1['Gender'] = df1['Gender'].map({'F':0,'M':1})     ##Mapping function for gender
df1.head()

#Handling Categorical Feature for Age
df1['Age'].unique()
df1['Age'] = df1['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})   #Mapping function for Age
df1.head()

#Fixing categorical City_category
df_city = pd.get_dummies(df1['City_Category'],drop_first=True)
df_city.head()

pd.concat([df1,df_city],axis=1)
df_city.head()

#Drop city Category
df1.drop('City_Category',axis=1,inplace=True)
df1.head()

#Missing Values
df1.isnull().sum()


#Focus on replacing missing values
df1['Product_Category_1'].unique()
df1['Product_Category_2'].unique()

df1['Product_Category_2'].value_counts()

#Replace Missing Values with Mode
df1['Product_Category_2'].mode()
df1['Product_Category_2'].mode()[0]
df1['Product_Category_2'] = df1['Product_Category_2'].fillna(df1['Product_Category_2'].mode()[0])
df1['Product_Category_2'].isnull().sum()

#Product_Category_3
df1['Product_Category_3'].unique()
df1['Product_Category_3'].value_counts()

#Replace Missing Values with Mode For Category_3
df1['Product_Category_3'].mode()
df1['Product_Category_3'].mode()[0]
df1['Product_Category_3'] = df1['Product_Category_3'].fillna(df1['Product_Category_3'].mode()[0])
df1['Product_Category_3'].isnull().sum()

#For Stay_In_Current_City_Years
df1['Stay_In_Current_City_Years'].unique()

df1['Stay_In_Current_City_Years'].str.replace('+','')
df1['Stay_In_Current_City_Years'] = df1['Stay_In_Current_City_Years'].str.replace('+','')
df1.head()
df1.info()

#Convert Object Into Integer
#df1['Stay_In_Current_City_Years']=df1['Stay_In_Current_City_Years'].astype(int)
#df1.info()
#df1['B']=df1['B'].astype(int)
#df1['C'] = df1['C'].astype(int)

#Visualisation

#sns.pairplot(df1)
sns.barplot('Age','Purchase',hue='Gender',data=df1)
#Purchasing of men is high then women

#Visualisation of Purchase with Occupation
sns.barplot('Occupation','Purchase',hue='Gender',data=df1)
sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df1)
sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df1)
sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df1)

#Feature Scaling
df1_test = df1[df1['Purchase'].isnull()]
df1[~df1['Purchase'].isnull()]

df1_train = df1[~df1['Purchase'].isnull()]

X = df1_train.drop('Purchase',axis=1)
X.head()
X.shape
y = df1_train['Purchase']
y
y.shape




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
      df1_test, y, test_size=0.33, random_state=42)

X_train.drop('Product_Id',axis=1,inplace=True)
X_test.drop('Product_Id',axis=1,inplace=True)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


##Train your model













