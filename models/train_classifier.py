import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import libraries
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    Input:
    database_filepath: Filepath for the db. [string]

    Output:
    X: Feature data. [dataframe]
    Y: Label data. [dataframe]
    category_names: Category names. [list of strings]

    Description:
    Load data from db and returns feature and label data and list of category names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster", engine)

    # Select feature
    X = df['message']

    # Select labels
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    
    # Select category names
    category_names = Y.columns.values
    
    return X, Y, category_names


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
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that creates feature based on verb presence in text
    """
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

def build_model():
    """
    Build a ML pipeline and optimize with GridSearch
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 150, 250]        
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=3, verbose=1, refit = True)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
    model: ML model. [model object]
    X_test: Dataframe containing test features. [dataframe]
    Y_test: Dataframe containing test labels. [dataframe]
    category_names: Category names. [list of strings]

    Output:
    Classification report and best parameters

    Description:
    Prints classification report and best parameters selected from GridSearch
    """
    y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print(classification_report(Y_test.iloc[i], y_pred[i]))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """
    Input:
    model: ML model. [model object]
    model_filepath: Filepath for where ML model will be saved. [string]

    Output:
    None

    Description:
    Saves ML model in pickle format.
    """
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

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()