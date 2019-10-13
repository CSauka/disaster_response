import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Line
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download("wordnet")
nltk.download("stopwords")

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words("english"):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine(r'sqlite:///..\data\DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # data for graph 1: Distribution of genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index.str.capitalize())
    annotations_graph_1 = ['{:,} Messages'.format(x) for x in genre_counts]

    # data for graph 2: Distribution of cqtegories
    category_data_vertical = df.iloc[:, -36:].sum().sort_values(ascending=True)
    category_counts = category_data_vertical.values
    category_labels = category_data_vertical.index.str.replace("_"," ").str.capitalize()
    annotations_graph_2 = ['{:,}'.format(x) for x in category_counts]

    # data for graph 3: Distribution of the number of cqtegories
    category_data_horizontal = df.iloc[:, -36:].sum(axis=1).value_counts()
    proporation_of_messages = category_data_horizontal.values / df.shape[0]
    no_of_message_categories = category_data_horizontal.index
    annotations_graph_3 = ['{:.2%}'.format(x) for x in proporation_of_messages]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #graph 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text=annotations_graph_1,
                    textposition='inside',
                    hoverinfo='skip',
                    marker_color='#b2ddf7'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Number of Messages",
                    'showline': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'showgrid': False,
                    'visible': False
                },
                'xaxis': {
                    'title': "Genre",
                    'type': "category",
                    'categoryorder': "category descending"
                }
            }
        },
        #graph 2
        {
            'data': [
                Bar(
                    x=category_counts,
                    y=category_labels,
                    orientation="h",
                    text=annotations_graph_2,
                    textposition='outside',
                    textfont={
                        'color': 'gray',
                        'size': 11
                        },
                    hovertemplate = 'Attributed to category <b>%{y}</b>:'
                                    '<br><b>%{x:,}</b> messages',
                    marker_color='#81d6e3'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message Category",
                    'type': "category",
                    'dtick': 1,
                    'automargin': True,
                    'showline': False,
                    'zeroline': True,
                },
                'xaxis': {
                    'title': "Number of Messages attributed to a Category",
                    'showline': True,
                    'zeroline': False,
                    'showticklabels': False,
                    'showgrid': False
                },
                'bargap': 0.2,
                'height': 800
            }
        },
        #graph 3
        {
            'data': [
                Bar(
                    x=proporation_of_messages,
                    y=no_of_message_categories,
                    orientation="h",
                    text=annotations_graph_3,
                    textposition='outside',
                    textfont={
                        'color': 'gray',
                        'size': 11
                        },
                    hovertemplate = 'Attributed to <b>%{y:.0f}</b> categories:'
                                    '<br><b>%{x:.2%}</b> of all messages',
                    marker_color='#00a7e1'
                )
            ],

            'layout': {
                'title': 'Distribution of the Number of Categories per Message',
                'yaxis': {
                    'title': "Number of Categories",
                    'showline': False,
                    'zeroline': False,
                    'type': "integer",
                    'dtick': 1,
                    'autorange': 'reversed',
                    'showline': False,
                    'zeroline': False,
#                    'type': "category",
#                    'dtick': 1,
#                    'automargin': True
                },
                'xaxis': {
                    'title': "Proportion of Messages",
                    'showline': True,
                    'zeroline': False,
                    'showticklabels': False,
                    'showgrid': False
                },
                'bargap': 0.2,
                'height': 800
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
