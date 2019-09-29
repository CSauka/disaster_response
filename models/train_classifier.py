import sys
import re
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV

import pickle

nltk.download("punkt")

def load_data(database_filepath):
    """load messages from database"""
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("messages", engine)[:1000]
    X = df["message"]
    Y = df.iloc[:, -36:]
    labels = df.columns[-36:]

    return X, Y, labels


def tokenize(text):
    """tokenizer to be used within the CountVectorizer"""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    return word_tokenize(text)


def build_model():
    pipeline = Pipeline([("vectorizer", CountVectorizer(tokenizer=tokenize)),
                     ("tfidf", TfidfTransformer()),
                     ("clf", MultiOutputClassifier(DecisionTreeClassifier()))])
    parameters = {"clf__estimator__max_depth": [5, 10, None],
                  "clf__estimator__min_samples_split": [2, 3, 4]}
    model = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """make and score predictions based on the given model"""
    Y_pred = model.predict(X_test)
#    for col in range(Y_pred.shape[1]):
#        print(category_names[col].upper())
#        print(classification_report(Y_test.values[:,col], Y_pred[:,col]))
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))
#    print(multilabel_confusion_matrix(Y_test.values, Y_pred))


def save_model(model, model_filepath):
    """save the trained model in a pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
