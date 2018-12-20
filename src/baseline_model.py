import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from nltk.tokenize import TweetTokenizer

from xml2df import XMLParser

data_dir = os.path.dirname(__file__) + '/../data'

question_train_filename = data_dir + '/questions_train.xml'
question_dev_filename = data_dir + '/questions_dev.xml'
answer_train_filename = data_dir + '/answers_train.xml'
answer_dev_filename = data_dir + '/answers_dev.xml'

def df_from_xml_file(filename):
    with open(filename, 'r') as content_file:
        content = content_file.read()

    xml = XMLParser(content)
    xml_df = xml.to_df()
    return xml_df

def make_data(df, vect, tmap):
    q_type = pd.DataFrame({
        'subject': df.RelQSubject,
        'question': df.RelQBody,
        'type': df.RELQ_FACT_LABEL,
    })
    q_type.question = q_type.question.fillna("")
    q_type.question = q_type.subject + "\t"  + q_type.question
    q_type = q_type.drop(['subject'], axis=1)

    x = vect.fit_transform(q_type.question)

    y = q_type.type.transform(lambda z: tmap[z])
    return x, y


tfidf = TfidfVectorizer()
types_map = {
    'Factual': 0,
    'Opinion': 1,
    'Socializing': 2,
}

question_train_df = df_from_xml_file(question_train_filename)
question_dev_df = df_from_xml_file(question_dev_filename)

x_train, y_train = make_data(question_train_df, tfidf, types_map)
x_val, y_val = make_data(question_dev_df, tfidf, types_map)

classifier = SVC()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_val)

print(y_val == y_pred)
