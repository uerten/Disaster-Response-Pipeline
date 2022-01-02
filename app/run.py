import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.corpus import stopwords

import re
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar, Pie
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

def tokenize(text):
    """
    Input:
    text: Collected message. [string]

    Output:
    tokens: List of strings containing normalized and stemmed tokens. [list of string]

    Description:
    This function normalize, tokenize and stem the texts.
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # list of all detected urls
    detected_urls = re.findall(url_regex, text)
    
    # replace all urls with "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize the text
    tokens = word_tokenize(text)

    # lemmatizer initation
    lemmatizer = WordNetLemmatizer()

    # iterate through each token and normalize case, lemmatize and remove white spaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# load data on Linux
# engine = create_engine('sqlite:///../data/DisasterResponse.db')

# load data on Windows
engine = create_engine('sqlite:///.\data\DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model on Linux
# model = joblib.load("../models/classifier.pkl")

# load model on Windows
model = joblib.load(r".\models\classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Plot category by counts and percentages
    category_counts = df.iloc[:,4:].sum(axis = 0).sort_values(ascending = False)
    category_names = category_counts.index.values

    # Pie chart frequency of words
    word_list = df['message'].str.lower().str.split().explode().reset_index(drop=True)
    top_words = word_list.loc[~word_list.isin(stopwords.words("english"))].value_counts()[:10]
    word_names = top_words.index.values
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text = genre_counts / genre_counts.sum(),
                    hoverinfo = 'y+text',
                    hovertemplate = "Percent: %{text:.1f}% | Counts: %{y}"
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    text = category_counts / category_counts.sum(),
                    hoverinfo = 'y+text',
                    hovertemplate = "Percent: %{text:.1f}% | Counts: %{y}"
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=word_names,
                    values=top_words,
                    #text = top_words / top_words.sum(),
                    hoverinfo = 'label+percent'
                    #hovertemplate = "Percent: %{text:.2f}% | Counts: %{value}"
                )
            ],

            'layout': {
                'title': 'Distribution of Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()