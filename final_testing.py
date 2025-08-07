import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from log_file import phase_1
logger = phase_1('final_testing')
import sys

def test_all_models():
    try:
        results = {}

        # Logistic Regression
        model_ = pickle.load(open("Logistic.pkl", 'rb'))
        temp = np.random.random((5, 2)).ravel()
        prediction = "Good Customer" if model_.predict([temp])[0] == 1 else "Bad Customer"
        results["Logistic Regression"] = prediction
        logger.info(f"Logistic Regression prediction: {prediction}")

        # KNN
        model_ = pickle.load(open("KNN_model.pkl", 'rb'))
        temp = np.random.random((5,2)).ravel()
        prediction = "Good Customer" if model_.predict([temp])[0] == 1 else "Bad Customer"
        results["KNN"] = prediction
        logger.info(f"KNN prediction: {prediction}")

        # Decision Tree
        model_ = pickle.load(open("DecisionTree_model.pkl", 'rb'))
        temp = np.random.random((5,2)).ravel()
        prediction = "Good Customer" if model_.predict([temp])[0] == 1 else "Bad Customer"
        results["Decision Tree"] = prediction
        logger.info(f"Decision Tree prediction: {prediction}")

        # Naive Bayes
        model_ = pickle.load(open("NaiveBayes_model.pkl", 'rb'))
        temp = np.random.random((5,2)).ravel()
        prediction = "Good Customer" if model_.predict([temp])[0] == 1 else "Bad Customer"
        results["Naive Bayes"] = prediction
        logger.info(f"Naive Bayes prediction: {prediction}")

        # Random Forest
        model_ = pickle.load(open("RandomForest_model.pkl", 'rb'))
        temp = np.random.random((5,2)).ravel()
        prediction = "Good Customer" if model_.predict([temp])[0] == 1 else "Bad Customer"
        results["Random Forest"] = prediction
        logger.info(f"Random Forest prediction: {prediction}")

        return results

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Prediction Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

