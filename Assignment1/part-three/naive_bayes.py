__all__ = ['read_reviews', 'create_dataframe', 'clean_review', 'read_test_file', 'NaiveBayesSentimentClassifier',
           'run_naive_bayes', 'run_test_file']

import os
import string
import pandas as pd
from collections import defaultdict
import math
from typing import Union


def read_reviews(path: str # The path to the folder containing the subdirectories 'pos' and 'neg'.
                ):
    """
    Given a path, reads the data from the two subdirectories 'pos' and 'neg'.

    Parameters:
        - path (str): The path to the folder containing the subdirectories 'pos' and 'neg'.

    Returns:
    - A list of reviews as strings
    """

    reviews = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf8") as f:
                review = f.read().strip()
                reviews.append(review)
    return reviews

def create_dataframe(path: str # The path to the folder containing the subdirectories 'pos' and 'neg'.
                    ):
    """
    Creates a pandas dataframe with two columns, Review (this is the review from the text file)
    and Label (this contains 'positive' or 'negative' if it came from the 'pos' or 'neg' subdirectory respectively).

    Parameters:
        - folder_path (str): The path to the folder containing the subdirectories 'pos' and 'neg'.

    Returns:
    - A pandas dataframe with two columns, Review and Label.
    """
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "neg")

    # Read in all the positive and negative reviews
    pos_reviews = read_reviews(pos_path)
    neg_reviews = read_reviews(neg_path)

    # Create a pandas dataframe with the reviews and labels
    reviews = pos_reviews + neg_reviews
    labels = ["positive"] * len(pos_reviews) + ["negative"] * len(neg_reviews)
    data = {"Review": reviews, "Label": labels}
    df = pd.DataFrame(data)
    return df

def clean_review(review: str # The review to be cleaned.
                ):
    """
    Cleans a review by converting text to lowercase, removing numbers and punctuation, stop words (specified in the function) and extra whitespace.

    Parameters:
        - review (str): The review to be cleaned.

    Returns:
    - A list of clean words.
    """

    # Defining stop words and punctuation
    stopwords = [
        'and', 'the', 'is', 'in', 'that', 'those', 'there', 'of', 'to', 'by', 
        'an', 'i', 'me', 'it', 'you', 'her', 'he', 'a', 'this', 'are', 'about', 
        'his', 'movie', 'so', 'or', 'what', 'film', 'for', 'as', 'they', 'them',
        'her', 'him', 'with', 'she', 'at', 'which', 'on', 'be', 'who', 'when', 'where'
    ]
    # Remove stop words and extra whitespace
    clean_words = [word for word in review.split() if word not in stopwords \
                   + list(string.punctuation) + list(string.digits)]
    return clean_words


def read_test_file(file_path: str # Path to the file to read
                     ) -> pd.Series:
    '''
    Reads a file given a path to the file where every line in the file is one review, 
    transforms the file and returns it as a pandas series.
    
    Parameters:
        - file_path (str): Path to the file to read
    
    Returns:
        - pandas.Series: A pandas series where each element is a review from the file
    '''
    
    # Open file and read lines into a list
    with open(file_path, 'r', encoding='utf-8') as f:
        reviews_list = f.readlines()

    # Strip whitespace from each line and create pandas series
    reviews = pd.Series([review.strip() for review in reviews_list])

    return reviews

class NaiveBayesSentimentClassifier:
    
    def __init__(self):
        self.prior_positive = 0.0
        self.prior_negative = 0.0
        self.word_count_positive = defaultdict(int)
        self.word_count_negative = defaultdict(int)
        self.vocab = set()
    
    def train(self, 
              df:pd.DataFrame, # A dataframe containing two columns: 'CleanReview' (contains lists of words) and 'Label' (contains either 'positive' or 'negative')
              alpha:float=1.0 # Smoothing parameter for Laplace smoothing (default 1.0).
             ):
        '''
        Trains a Naive Bayes sentiment classifier on a given dataframe.
        
        Parameters:
            - df (pd.DataFrame): A dataframe containing two columns:
                                 'CleanReview' (contains lists of words from review) and
                                 'Label' (contains either 'positive' or 'negative')
            - alpha (float): Smoothing parameter for Laplace smoothing (default 1.0).
        
        Returns:
            None
        '''
        
        # Splitting up positive and negative reviews
        positive_reviews = df[df['Label'] == 'positive']
        negative_reviews = df[df['Label'] == 'negative']
        
        # Calculating the prior probabilites of each class
        self.prior_positive = len(positive_reviews) / len(df)
        self.prior_negative = len(negative_reviews) / len(df)
        
        # Creating a positive word count and developing the vocabulary
        for _, row in positive_reviews.iterrows():
            for word in row['CleanReview']:
                self.word_count_positive[word] += 1
                self.vocab.add(word)

        # Creating a negative word count and developing the vocabulary
        for _, row in negative_reviews.iterrows():
            for word in row['CleanReview']:
                self.word_count_negative[word] += 1
                self.vocab.add(word)
        
        self.prob_word_given_positive = defaultdict(float)
        self.prob_word_given_negative = defaultdict(float)
        
        total_count_positive = sum(self.word_count_positive.values())
        total_count_negative = sum(self.word_count_negative.values())
        
        # Calculating the probabilities for each word given a class and Laplace Smoothing
        # P(word|label) = Number of occurences of a word for a class / Total number of occurances for the class
        for word in self.vocab:
            self.prob_word_given_positive[word] = (self.word_count_positive[word] + alpha) / (total_count_positive + alpha*len(self.vocab))
            self.prob_word_given_negative[word] = (self.word_count_negative[word] + alpha) / (total_count_negative + alpha*len(self.vocab))

    def predict(self, reviews:pd.Series) -> pd.Series:
        '''
        Predicts the sentiment for a given dataframe of reviews.
        
        Parameters:
            - df (pd.DataFrame): A dataframe containing one column: 'CleanReview' (contains lists of words) and 'Label' (optional).
        
        Returns:
            A dataframe with two columns: 'Prediction' (contains either 'positive' or 'negative') and 'Probability' (contains the probability of the prediction).
        '''
        
        # Create a results dataframe
        predictions = []
        probs_positive = []
        probs_negative = []
        
        for review in reviews:
            
            # Initialising the probabilities
            prob_positive = math.log(self.prior_positive)
            prob_negative = math.log(self.prior_negative)
            
            # Add probabilities for each word        
            for word in review:
                if word in self.vocab:
                    prob_positive += math.log(self.prob_word_given_positive[word])
                    prob_negative += math.log(self.prob_word_given_negative[word])
            
            # Add Probabilites
            probs_positive.append(prob_positive)
            probs_negative.append(prob_negative)
            
            # If it's more likely to be positive, predict positive else predict negative
            if prob_positive > prob_negative:
                predictions.append('positive')
                
            else:
                predictions.append('negative')
        
        # Add the predictions and probabiities
        self.test_reviews = reviews
        self.predictions = pd.Series(predictions)
        self.probs_positive = pd.Series(probs_positive)
        self.probs_negative = pd.Series(probs_negative)
        return self.predictions 
    
    def evaluate(self, labels:pd.Series) -> float:
        '''
        Evaluates the accuracy of the sentiment predictions.

        Parameters:
            - No parameters as it uses df_results

        Returns:
            - accuracy (float): The accuracy of the sentiment predictions
        '''
        self.labels = labels
        
        total_predictions = len(self.predictions)

        correct_predictions = (self.predictions == self.labels).sum()

        self.accuracy = correct_predictions / total_predictions

        return self.accuracy


def run_naive_bayes(train_file, test_file):
    print('Reading in the train and test data\n-')
    df_train = pd.read_pickle(train_file).reset_index(drop=True)
    df_test = pd.read_pickle(test_file).reset_index(drop=True)
    
    print('Creating classifier\n-')
    nb = NaiveBayesSentimentClassifier()
    
    print('Training the classifier\n-')
    nb.train(df_train)
    
    print('Making predictions\n-')
    nb.predict(df_test['CleanReview'])
    
    print('Evaluating predictions\n-')
    nb.evaluate(df_test['Label'])
    
    print(f'All done! You achieved {nb.accuracy * 100:.2f}% accuracy!\n\n')
    return nb

def run_test_file(test_file, nb):
    reviews = read_test_file(test_file)
    nb.predict(reviews)
    return nb
