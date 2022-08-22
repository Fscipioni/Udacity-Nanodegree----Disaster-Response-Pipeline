import sys
import nltk

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# import statements
import pandas as pd
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """
    Load the data from the database_filepath.

    INPUT
        database_filepath --> The database containing the data

    OUTPUT
        X --> Dataframe with the messages
        Y --> Dataframe with the categories
        category_names --> list of all categories
    """

    # load the data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # Create the dataframes with the messages and the categories
    # Extract the list of all categories
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the input text

    OUTPUT
        clean_tokens --> tokenized text
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatization and clean up of text
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build and return the model
    """

    # Define the pipline:
    #   CountVectorizer to apply the tokenize function to the text
    #   TfidfTransformer to transform a count matrix to a normalized tf or tf-idf representation
    #   MultiOutputClassifier(RandomForestClassifier()) to apply the random forest classifier to multiple
    #   target variables

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                         ])

    # Search for the best parameters
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'clf__estimator__min_samples_split': [2, 3]
                  }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)
    
    print('finished')

    return model

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))

    return loaded_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance

    INPUT
        model --> The model to evaluate
        X_test, Y_test --> Datasets for testing the model
        category_names --> list of all categories

    """

    # Evaluate the model performance
    Y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        # print('Category: {} '.format(col))
        print(type(Y_pred))
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test[:, i], Y_pred[:, i])))


def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    INPUT
        model --> the model to save
        model_filepath --> the model file
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
