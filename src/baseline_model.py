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

def df_from_xml_file(filename):
    """Make dataframe from xml file(not recursive)"""
    with open(filename, 'r') as content_file:
        content = content_file.read()

    xml = XMLParser(content)
    xml_df = xml.to_df()
    return xml_df

def make_vocab(docs, tokenizer):
    """Crates list with unique words from given documents using tokenizer"""
    vocab = set()
    for doc in docs:
        tokenized_doc = tokenizer.tokenize(doc)
        vocab.update(tokenized_doc)
    return list(vocab)

def make_data(df, vectorizer, class_map):
    """Compact and transform the dataframe needed for classification.
    Returns:
        x - 2d nparray with features from tfidf vectorizer
        y - 1d nparray with classes, transformed to numericals from class_map
        data - dataframe with only the needed features for classification
    """
    data = pd.DataFrame({
        'id': df.THREAD_SEQUENCE,
        'subject': df.RelQSubject,
        'question': df.RelQBody,
        'type': df.RELQ_FACT_LABEL,
    })

    data.question = data.question.fillna("")
    data.question = data.subject + "\t"  + data.question
    data = data.drop(['subject'], axis=1)

    x = vectorizer.fit_transform(data.question)

    if class_map == None:
        return x, None, data

    y = data.type.transform(lambda z: class_map[z])
    return x, y, data

def print_formated_output(predict_df):
    """Print predictions from given dataframe as expected from the grader"""
    for i in range(predict_df.shape[0]):
        print("{}\t{}".format(predict_df.id[i], predict_df.pred[i]))

def print_score(y_test, y_pred):
    """Print accuracy, f1_score and confusion_matrix to stdout"""
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("ACC: {}".format(accuracy))
    print("F1: {}".format(f1))
    print(conf_matix)


if __name__ == "__main__":
    data_dir = os.path.dirname(__file__) + '/../data'

    question_train_filename = data_dir + '/questions_train.xml'
    question_val_filename = data_dir + '/../starting_kit/input/ref/input_questions.xml'
    question_test_filename = data_dir + '/questions_dev.xml'

    # Load all available data
    question_train_df = df_from_xml_file(question_train_filename)
    question_val_df = df_from_xml_file(question_val_filename)
    question_test_df = df_from_xml_file(question_test_filename)

    # Create vocabulary, vectorizer and map for classes
    tokenizer = TweetTokenizer()
    docs = question_train_df.RelQSubject + "\t"  + question_train_df.RelQBody
    vocab = make_vocab(docs.fillna(""), tokenizer)
    tfidf = TfidfVectorizer(stop_words='english',
                            norm='l2',
                            tokenizer=tokenizer.tokenize,
                            vocabulary=vocab)

    types_map = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}

    # Divide the data to train and test, also vectorized it using tf-idf
    train_df = question_train_df.append(question_val_df)
    x_train, y_train, df_train = make_data(train_df, tfidf, types_map)
    x_test, _, df_test = make_data(question_test_df, tfidf, None)

    # Train classifier and predict on test data
    classifier = SVC(C=1, gamma='auto', kernel='linear')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_pred_df = pd.concat([df_test.id,
                           df_test.question,
                           pd.Series(y_pred, name='pred')],
                          axis=1, sort=False)

    # Print the predictions on stdout in the format requested by the grader
    print_formated_output(y_pred_df)
