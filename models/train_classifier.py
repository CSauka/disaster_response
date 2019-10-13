import sys
import re
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """loads messages from the database"""
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("messages", engine)
    X = df["message"]
    Y = df.iloc[:, -36:]
    labels = df.columns[-36:]

    return X, Y, labels


def tokenize(text):
    """specifies tokenizer to be used within the CountVectorizer"""
    # stopwords are imported inside the function to prevent a pickling error
    # when running grid search on multipe CPUs
    # (pickle.PicklingError: args[0] from __newobj__ args has the wrong class)
    from nltk.corpus import stopwords
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = word_tokenize(text)
    compact = [tok for tok in tokens if tok not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in compact]

    return lemmed


def build_model():
    """compiles and returns a model consisting of a sklearn pipeline and
    corresponding grid search parameters
    """
    pipeline = Pipeline([("vectorizer", CountVectorizer(tokenizer=tokenize)),
                         ("tfidf", TfidfTransformer()),
                         ("clf", MultiOutputClassifier(AdaBoostClassifier()))])
    parameters = {"clf__estimator__n_estimators": [25, 50],
                  "clf__estimator__learning_rate": [0.7, 0.8]}
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-2, cv=5,
                         verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """makes and scores predictions based on the given model"""
    Y_pred = model.predict(X_test)
    for col in range(Y_pred.shape[1]):
        print(category_names[col].upper())
        print(classification_report(Y_test.values[:,col], Y_pred[:,col]))


def save_model(model, model_filepath):
    """saves the trained model into pickle file"""
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
        print("Parameters chosen during grid search:")
        print(model.best_params_)

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
