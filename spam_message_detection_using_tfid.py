#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.naive_bayes import MultinomialNB
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# # Loading the data

# In[2]:


df = pd.read_csv('SPAM text message 20170820 - Data.csv')
df.head(10)


# # checking the data

# In[3]:


df.columns


# In[4]:


df.shape


# #checking if any null value exist in dataset

# In[5]:


df.isnull().sum()
df.isna().sum()


# # Checking if HTML tags exist

# In[6]:


#checking HTML tag....

for i in df['Message'].values:
    if(len(re.findall('<.*?>', i))):
        print(i)
        print('\n')
    
    
    
    


# # Loading the english stop words

# In[7]:


stop = set(stopwords.words('english'))
print(stop)
print(type(stop))


# # droping the duplicates values

# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.shape


# In[10]:


df['Category'].value_counts()


# # Function to clean punctuation

# In[11]:


def cleanpunc(sentences):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentences)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# # Loading the snowball stemmer

# In[12]:


from nltk.stem import SnowballStemmer
sno = SnowballStemmer('english')


# In[ ]:





# In[13]:


str1=' '
s=' '
i = 0
final_string = []
for wor in df['Message'].values:
    fil_wor = []
    for w in wor.split():
        for cleanedwords in cleanpunc(w).split(): #cleaning punctuation
            if(cleanedwords.isalpha() and len(cleanedwords)>2): #checking value is alpha numeric or not and we know adjective size is greater than 2
                if(cleanedwords.lower() not in stop):
                    s=(sno.stem(cleanedwords.lower())).encode('utf8') #applying stemmer and converting the character to lowercase
                    fil_wor.append(s)
    str1 = b" ".join(fil_wor) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1


# In[14]:


#copying the column to exixsting dataset
df['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review


# In[15]:


df


# # Visualizing the spam and ham words

# In[16]:


spam_words = ' '.join(list(df[df['Category'] == 'spam']['Message']))
spam_wc = WordCloud(width = 512, height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.show()


# In[17]:


ham_words = ' '.join(list(df[df['Category'] == 'ham']['Message']))
ham_wc = WordCloud(width = 512, height = 512).generate(ham_words)
plt.figure(figsize = (10, 8))
plt.imshow(ham_wc)
plt.show()


# In[18]:


df.columns


# In[19]:


df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
df


# In[20]:


#droping the Category  columns
df.drop('Category', axis=1, inplace=True)


# In[21]:


#distributing the dataset into feature and label
x = df['CleanedText']
y = df['label']


# In[22]:


#text is represented as the tfid
tfid = TfidfVectorizer(ngram_range=(1,2))
x = tfid.fit_transform(x)


# # Spliting the dataset into train and test feature and labels

# In[23]:



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# # Applying Multinomial classification on traing data

# In[24]:


classifier = MultinomialNB()
classifier.fit(x_train, y_train)


# # checking the accuracy

# In[25]:



pred = classifier.predict(x_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix:\n',confusion_matrix(y_test, pred))
print()
print('Accuracy : ',accuracy_score(y_test, pred))


# # Applying KNN  and checking accuracy

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)


# In[27]:


pred = neigh.predict(x_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix:\n',confusion_matrix(y_test, pred))
print()
print('Accuracy : ',accuracy_score(y_test, pred))


# # Applying SVM and checking its accuracy

# In[28]:


from sklearn import svm
svm = svm.SVC()
svm.fit(x_train, y_train)


# In[29]:


pred = svm.predict(x_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix:\n',confusion_matrix(y_test, pred))
print()
print('Accuracy : ',accuracy_score(y_test, pred))

