# Disaster Response Pipelines


### Table of Contents

1. [Project Description](#description)
2. [Installation and Libraries](#installation)
3. [List or Files](#files)
4. [Project Pipelines and Web App](#pipeline)
5. [Running Instructions](#instructions)


### Project Description<a name = "description"></a>

This project is part of the [Udacity](https://www.udacity.com/) Data Scientist Nanodegree Program.

The goal of the project is to analyze disaster data from [Appen](https://www.figure-eight.com/) (formally Figure 8) to build a model for an API that classifies disaster messages.
The analyzed data set contains real messages that were sent during disaster events. The analysis consists in the application of a machine learning pipeline to categorize these events so that a messages is sent to an appropriate disaster relief agency.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.


### Installation and Libraries <a name="installation"></a>

The code should run with no issues using Python versions 3.* 
The libraries used are are:

- pandas
- re
- sys
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- json


### List of Files<a name = "files"></a>

- app
	| - template
	| 	|- master.html  # main page of web app
	| 	|- go.html  # classification result page of web app
	|- run.py  # Flask file that runs app

- data
	|- disaster_categories.csv  # data to process 
	|- disaster_messages.csv  # data to process
	|- process_data.py
	|- InsertDatabaseName.db   # database to save clean data to

- models
	|- train_classifier.py
	|- classifier.pkl  # saved model 

- README.md


### Project Pipelines and Web App<a name = "pipeline"></a>

The project consists of three componants:

he project consists of three componants:

1. **ETL Pipeline**
	- Python script 'process_data.py':

		- Loads the messages and categories datasets
		- Merges the two datasets
		- Cleans the data
		- Stores it in a SQLite database

2. **ML Pipeline**
	- Python script 'train_classifier.py':

		- Loads data from the SQLite database
		- Splits the dataset into training and test sets
		- Builds a text processing and machine learning pipeline
		- Trains and tunes a model using GridSearchCV
		- Outputs results on the test set
		- Exports the final model as a pickle file

3. **Flask Web App**
	- The web app allows an emergency worker to input a new emergency ir disaster message and get classification results in several categories. 
	  The web app also displays visualizations of the data.


### Running Instructions <a name="instructions"></a>

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
