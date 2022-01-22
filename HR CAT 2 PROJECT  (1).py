#!/usr/bin/env python
# coding: utf-8

# # TEAM - 11

# ## CAT 2

# ### TO PREDICT THE REASON WHY THE EMPLOYEES LEFT THE ORGANIZATION WITH DATA VISUALIZATION ANH MACHINE LEARNING MODELS

# In[38]:


# IMPORTING MODULES:

import pandas  
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[41]:


# LOADING THE DATASET:

data=pandas.read_csv('C:/Users/Krishnadev/Downloads/HR_comma_sep.csv')
data


# In[42]:


data.head()


# In[4]:


data.tail()


# In[12]:


data.info()


# In[14]:


# DATA INSIGHTS:

left = data.groupby('left')
left.mean()


# In[13]:


data.describe()


# # DATA VISUALIZATION:

# In[11]:


## EMPLOYEES LEFT:

left_count=data.groupby('left').count()
plt.bar(left_count.index.values, left_count['satisfaction_level'])
plt.xlabel('Employees Left Company')
plt.ylabel('Number of Employees')
plt.show()


# In[15]:


data.left.value_counts()


# In[16]:


## NUMBER OF PROJECTS:

num_projects=data.groupby('number_project').count()
plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
plt.xlabel('Number of Projects')
plt.ylabel('Number of Employees')
plt.show()


# In[17]:


## TIME SPENT IN COMPANY:

time_spent=data.groupby('time_spend_company').count()
plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
plt.xlabel('Number of Years Spend in Company')
plt.ylabel('Number of Employees')
plt.show()


# In[18]:


## SUBPLOTS USING SEABORN:

features=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','Departments ','salary'] 
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features): 
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data)
    plt.xticks(rotation=90)
    plt.title("No. of employee")


# In[32]:


list(enumerate(features))


# In[27]:


fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='left')
    plt.xticks(rotation=90)
    plt.title("No. of employee")


# # BUILDING A MODEL

# In[20]:


#PREPROCESSING

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['Departments ']=le.fit_transform(data['Departments '])


# In[21]:


#Spliting data into Feature and
X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Departments ', 'salary']]
y=data['left']


# In[22]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test


# # GRADIANT BOOSTER TREE

# In[23]:


#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)


# In[24]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))


# ## KNN MODEL

# In[33]:




#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


# In[34]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))


# # ADAPTIVE BOOSTING (ADA BOOSTER) MODEL

# In[35]:




from sklearn.ensemble import AdaBoostClassifier
#Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=45,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


# In[36]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))

