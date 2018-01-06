import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

def plot_graph_as_pdf_DT(clf):
    dot_data = tree.export_graphviz(clf, out_file=None,class_names='class', filled=True, rounded=True,special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("shrooms") 


#To transform the datFrame to format returned by load_iris
def load_mush(pandas_df):
    features = dict()
    for i in range(0,len(pandas_df)):
        x = []
        for row in pandas_df:
            if str(row) == "class" and pandas_df[row].get(i)!=None:
                features.setdefault('target',[]).append(pandas_df[row][i])
            elif pandas_df[row].get(i)!=None:
                x.append(pandas_df[row][i])
        if len(x)>0:
            features.setdefault('features',[]).append(x)
    features['target'] = np.asarray(features['target'])
    features['features'] = np.asarray(features['features'])
    return features

def decisionTree():
    pandas_df = pd.read_csv('mushrooms.csv')

    #Label encoding is similar to StringIndexer in pyspark
    pandas_df =  pandas_df.apply(preprocessing.LabelEncoder().fit_transform)

    train, test = train_test_split(pandas_df, test_size=0.2)

    mush = load_mush(train)
    clf = tree.DecisionTreeClassifier()
    clf.fit(mush['features'],mush['target'])

    test_mush = load_mush(test)
    predictions = clf.predict(test_mush['features'])
    print accuracy_score(test_mush['target'], predictions)


def logisticRegression():
    from sklearn import linear_model

    pandas_df = pd.read_csv('mushrooms.csv')

    #Label encoding is similar to StringIndexer in pyspark
    pandas_df =  pandas_df.apply(preprocessing.LabelEncoder().fit_transform)

    train, test = train_test_split(pandas_df, test_size=0.2)

    mush = load_mush(train)
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(mush['features'],mush['target'])

    test_mush = load_mush(test)
    predictions = logreg.predict(test_mush['features'])
    print accuracy_score(test_mush['target'], predictions)  

logisticRegression()
decisionTree()