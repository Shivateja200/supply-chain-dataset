#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


import zipfile
import os
# Now, we have to unzip it
with zipfile.ZipFile('datacosupplychainDataset.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('Supply_chain_data')
print(os.listdir('Supply_chain_data'))


# In[7]:


# Our encoding needs to be change, so we change it to 'latin1' encoding to read the CSV file
file_path = 'Supply_chain_data/DataCoSupplyChainDataset.csv'  # Adjust file name if necessary
data = pd.read_csv(file_path, encoding='latin1')

data.head()


# In[9]:


data.info()
data.describe()
data.isnull().sum()


# In[11]:


# Checking missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Drop 'Product Description' as it is completely empty
data = data.drop(columns=['Product Description'])

# Handle other missing values, unkown instead of null
data['Customer Lname'] = data['Customer Lname'].fillna('Unknown')
data['Customer Zipcode'] = data['Customer Zipcode'].fillna('Unknown')
data['Order Zipcode'] = data['Order Zipcode'].fillna('Unknown')


# In[13]:


# Change fomat of some clumns date columns to datetime
data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'], format='%m/%d/%Y %H:%M')
data['shipping date (DateOrders)'] = pd.to_datetime(data['shipping date (DateOrders)'], format='%m/%d/%Y %H:%M')


# In[24]:


file_path = 'C:\Users\shiva\Downloads\DataCoSupplyChainDataset.csv'
data.to_csv(file_path, index=False)
print(f"Dataset saved successfully as '{file_path}'")


# In[26]:


# Summary statistics
print(data.describe())


# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of Sales
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], bins=30, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# In[30]:


# Scatter plot between Sales and Benefit per order
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sales', y='Benefit per order', data=data)
plt.title('Sales vs Benefit per Order')
plt.xlabel('Sales')
plt.ylabel('Benefit per Order')
plt.show()


# In[32]:


# Count plot of Late Delivery Risk by Shipping Mode
plt.figure(figsize=(12, 8))
sns.countplot(x='Late_delivery_risk', hue='Shipping Mode', data=data)
plt.title('Late Delivery Risk by Shipping Mode')
plt.xlabel('Late Delivery Risk')
plt.ylabel('Count')
plt.legend(title='Shipping Mode')
plt.show()


# In[34]:


print(data['Delivery Status'].value_counts())


# In[36]:


def generate_delivered_date(row):
    shipping_date = row['shipping date (DateOrders)']
    shipping_mode = row['Shipping Mode']

    # Initialize delivered date as NaN
    delivered_date = np.nan

    # Set delivered date only if delivery status is "Shipping on time"
    if row['Delivery Status'] == 'Shipping on time':
        if shipping_mode == 'Standard Class':
            delivered_date = shipping_date + pd.Timedelta(days=np.random.randint(1, 10))
        elif shipping_mode == 'Second Class':
            delivered_date = shipping_date + pd.Timedelta(days=np.random.randint(1, 5))
        elif shipping_mode == 'Same Day':
            delivered_date = shipping_date + pd.Timedelta(hours=np.random.randint(1, 12))
        elif shipping_mode == 'First Class':  # Assuming First Class is also a mode
            delivered_date = shipping_date + pd.Timedelta(days=np.random.randint(1, 3))

    return delivered_date

# Apply the function row-wise to generate delivered dates
data['Delivered Date'] = data.apply(generate_delivered_date, axis=1)

# Display the updated DataFrame
print(data)


# In[38]:


print(data['Delivered Date'].value_counts())


# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'data' is my DataFrame with the required columns
# Feature selection
X = data[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk']]
y = data['Delivery Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))


# In[42]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Feature selection
X = data[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk']]
y = data['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


# In[44]:


from sklearn.linear_model import LogisticRegression

X = data[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk']]
y = (data['Delivery Status'] == 'Shipping on time').astype(int)  # Convert to binary label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print(classification_report(y_test, y_pred))


# In[ ]:




