# Disaster Response Pipeline
Udacity Data Science Nanodegree Program Project-2

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Results](#results)
5. [Licensing](#licensing)

## Installation <a name="installation"></a>

1. Code runs Anaconda distribution for Python 3.9 and requires the libraries below:
    1. `sqlalchemy`
    2. `nltk`
    3. `plotly`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

## Project Motivation <a name="motivation"></a>

Second project of Udacity Data Science Nanodegree Program. Three main topics are covered in this project:
1. Create ETL pipeline to extract, clean, tokenize and load data to ML pipeline
2. Create ML pipeline to extract feature with estimator and transformer and build ML model. Optimize hyper-parameters with GridSearch
3. Build a Flask web app to display results

## File Description <a name="files"></a>

1. **data/process_data.py:** Data cleaning pipeline which extracts data from csv files, merges and cleans datasets and then stores it in SQLite database.
2. **model/train_classifier.py:** Machine Learning pipeline which loads data from SQLite database, tokenize and splits the data into training and test datasets. 
3. **app:** Flask app which contains dashboard and classification user interface.
4. **data:** This folder contains samples messages and categories datasets in csv format.

## Results <a name="results"></a>
soon

## Licensing <a name="licensing"></a>
Must give credit to [Appen](https://appen.com/) for the data they provide.
