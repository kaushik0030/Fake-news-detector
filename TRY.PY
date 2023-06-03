import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import tokenize
token_space = tokenize.WhitespaceTokenizer()
# precision = accuraccy



df_Fake = pd.read_csv("C:\\Users\\palla\\OneDrive\\Desktop\\PYTHON\\Fake.csv")
df_true = pd.read_csv("C:\\Users\\palla\\OneDrive\\Desktop\\PYTHON\\True.csv")

df_Fake.head()
df_true.head(5)

df_Fake["class"] = 0 # adds a new column in the dataframe
df_true["class"] = 1  # adds a new column in the dataframe

df_Fake.shape, df_true.shape


# taking the last 10 rows for testing purpose(manual) fake 
df_Fake_manual_testing = df_Fake.tail(10)  #new manual_testing dataframe object is created  for fake dataset
for i in range(23480,23470,-1):
    #the original data is directly modified 
    # the drop function is used to remove i'th row
    df_Fake.drop([i], axis = 0, inplace = True)
    
# taking the last 10 rows for testing purpose(manual)  true
df_true_manual_testing = df_true.tail(10)   #new manual_testing dataframe object is created for true dataset
for i in range(21416,21406,-1): 
    #the original data is directly modified 
    # the drop function is used to remove i'th row
    df_true.drop([i], axis = 0, inplace = True) 

df_Fake.shape, df_true.shape # returns the dimensions of the dataframe

df_Fake_manual_testing["class"] = 0   # adds a new column in the dataframe
df_true_manual_testing["class"] = 1   # adds a new column in the dataframe

df_true_manual_testing.head(10)

# both the files are merged
df_manual_testing = pd.concat([df_Fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

# both the files are merged
df_manual_testing = pd.concat([df_Fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

# merger
df_merge = pd.concat([df_Fake, df_true], axis =0 )
df_merge.head(10)

df_merge.columns

df = df_merge.drop(["title", "subject","date"], axis = 1)

# This line of code checks for the sum of missing values (null values) in each column of the Pandas DataFrame "df".
df.isnull().sum()

df = df.sample(frac = 1)

df.head()

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

df.columns

df.head()

# creating a function to process the texts for data cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

#remove stopwords
stop = stopwords.words('english')
df_merge['text'] = df_merge['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# category of news eg political news , world news
print(df_merge.groupby(['subject'])['text'].count())
df_merge.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

# show the categories of news ie fake news and true news
print(df_merge.groupby(['class'])['text'].count())
df_merge.groupby(['class'])['text'].count().plot(kind="bar")
plt.show()


df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)

LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

DT.score(xv_test, y_test)

print(classification_report(y_test, pred_dt))

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)

GBC.score(xv_test, y_test)

print(classification_report(y_test, pred_gbc))

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)

RFC.score(xv_test, y_test)

print(classification_report(y_test, pred_rfc))

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))

news = str(input())
manual_testing(news)

news = str(input())
manual_testing(news)

news = str(input())
manual_testing(news)