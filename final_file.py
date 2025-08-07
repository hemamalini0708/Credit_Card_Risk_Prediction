import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from log_file import phase_1
import pickle
import matplotlib.pyplot as plt
logger = phase_1("final_file")

def f_m(final_X_train, final_y_train, final_X_test, final_y_test):
    try:
        # Logistic Regression
        log_reg = LogisticRegression()
        log_reg.fit(final_X_train, final_y_train)
        log_pred = log_reg.predict(final_X_test)
        with open('Logistic.pkl', 'wb') as f_:
            pickle.dump(log_reg, f_)
        logger.info("Logistic Regression model saved successfully")

        # KNN
        knn = KNeighborsClassifier()
        knn.fit(final_X_train, final_y_train)
        knn_pred = knn.predict(final_X_test)
        with open('KNN_model.pkl', 'wb') as f_:
            pickle.dump(knn, f_)
        logger.info("KNN model saved successfully")

        # Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(final_X_train, final_y_train)
        dt_pred = dt.predict(final_X_test)
        with open('DecisionTree_model.pkl', 'wb') as f_:
            pickle.dump(dt, f_)
        logger.info("Decision Tree model saved successfully")

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(final_X_train, final_y_train)
        nb_pred = nb.predict(final_X_test)
        with open('NaiveBayes_model.pkl', 'wb') as f_:
            pickle.dump(nb, f_)
        logger.info("Naive Bayes model saved successfully")

        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(final_X_train, final_y_train)
        rf_pred = rf.predict(final_X_test)
        with open('RandomForest_model.pkl', 'wb') as f_:
            pickle.dump(rf, f_)
        logger.info("Random Forest model saved successfully")

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")
