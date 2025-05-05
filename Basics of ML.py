import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/ADMIN/Downloads/supermarket_sales - Sheet1.csv")#Reading the File

print(df)#Prints whole Data
print(df.info())#Prints information about the Data (Meta Data)
print(df.columns)#Prints all the attributes of the data
print(df.isnull().sum())#Gives null values in the data with data type
print(df.dtypes)#Also for data types

num_coloumn = df[[
 'Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income',
       'Rating']]
print(num_coloumn)


corr = num_coloumn.corr()#Finding Correlation
print(corr)

unique = df["gross margin percentage"].unique()
print(unique)

sns.heatmap(data=corr,annot=True,fmt=".2f")
plt.show()

print(df.columns)#Mentiones all the Attributes of the tables

df["Date"]=pd.to_datetime(df["Date"])
df['Time']=pd.to_datetime(df['Time'],format="%H:%M").dt.time
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()

# Darop
df.drop(columns=["Invoice ID","Time","Tax 5%","gross margin percentage"],inplace=True)

# label encoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cc = ['Branch', 'City', 'Customer type', 'Gender','Product line',"Payment"]
for i in cc:
    df[i]=le.fit_transform(df[i])
    
print(cc)
print(df.dtypes)

df["Sales"] = df['Quantity']

# feature Engeenering
df["sales1"] = df.groupby("Product line")["Sales"].shift(1)
df["sales2"] = df.groupby("Product line")["Sales"].shift(2)

# feature scaling --
X = df.drop(columns=["Date","Sales","Weekday"],axis=1)
y = df["Sales"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

print(x_test)

from sklearn.ensemble import RandomForestRegressor

RR = RandomForestRegressor()
RR.fit(x_train,y_train)

# model Evalution  -
y_pred = RR.predict(x_test)
print(y_pred)

# Find Accuracy -
from sklearn.metrics import r2_score

Accuracy = r2_score(y_pred,y_test)

print(Accuracy)

# Product Recommandation with pridiction

predicted_demand = RR.predict(sc.transform(X))
df['Predicted_Demand'] = predicted_demand
product_demand = df.groupby('Product line')['Predicted_Demand'].sum().reset_index()
product_demand = product_demand.sort_values(by='Predicted_Demand', ascending=False)
top_products = product_demand.head(5)
print("Top Recommended Products:")
print(top_products)

import matplotlib.pyplot as plt

# Plot the top-N products
plt.figure(figsize=(10, 6))
plt.barh(top_products['Product line'], top_products['Predicted_Demand'], color='skyblue')
plt.xlabel('Total Predicted Demand')
plt.title('Top Recommended Products by Demand')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()