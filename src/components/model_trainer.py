import os, sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "Support Vector Classifier": SVC(),
                "K-nearest Classifier": KNeighborsClassifier()
            }



            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # To get the best model score 
            best_model_score = max(sorted(model_report.values()))

            # To get best model from dict
            best_model_name = list(model_report.keys()) [list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("Best model not found")
            logging.info("Best model found on train and test set")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy



        except Exception as e:
            raise CustomException(e, sys)