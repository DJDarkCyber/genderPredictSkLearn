import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


names = pd.read_csv("data/name_gender.csv")
names = names.dropna()
names = names.values[:, :2]

TRAIN_SPLIT = 0.8

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],
        'first2-letters': name[0:2],
        'first3-letters': name[0:3],
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

features = np.vectorize(features)
X = features(names[:, 0])
y = names[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=437)

vectorizer = DictVectorizer()
vectorizer.fit(X_train)

clf = DecisionTreeClassifier()
clf.fit(vectorizer.transform(X_train), y_train)

def predictGenders(nameList):
    for names in nameList:
        if clf.predict(vectorizer.transform(features([names])))[0] == "M":
            print(f"{names} is a > Male")
        else:
            print(f"{names} is a > Female")
while True:
    usrNames = input("Enter the name to predict > ").split(',')
    predictGenders(usrNames)