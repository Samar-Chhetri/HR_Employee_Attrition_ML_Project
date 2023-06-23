import os, sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_nominal_columns = ['Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'OverTime']
            categorical_ordinal_column = ['BusinessTravel']

            numerical_columns = ['DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction','HourlyRate','JobInvolvement',
                                 'JobLevel', 'JobSatisfaction', 'MonthlyIncome','MonthlyRate','NumCompaniesWorked', 'PercentSalaryHike',
                                 'PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                                 'WorkLifeBalance','YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion', 'YearsWithCurrManager']
            
            cat_nominal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('ss', StandardScaler(with_mean=False))
            ])

            cat_ordinal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('oe', OrdinalEncoder(categories=[['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']])),
                ('ss', StandardScaler(with_mean=False))
            ])

            num_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='median')),
                ('ss', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('cat_nominal_pipeline', cat_nominal_pipeline, categorical_nominal_columns),
                ('cat_ordinal_pipeline', cat_ordinal_pipeline, categorical_ordinal_column),
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor

            
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = ['Attrition']

            input_feature_train_df = train_df.drop(columns= target_column_name)
            target_feature_train_df = train_df['Attrition']

            input_feature_test_df = test_df.drop(columns= target_column_name)
            target_feature_test_df = test_df['Attrition']

            logging.info("Applying preprocessor object on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            target_feature_test_arr = le.transform(target_feature_test_df)
            
            # Using SMOTE to upsample minority class
            smote = SMOTE()
            input_feature_train_arr_balanced, target_feature_train_arr_balanced = smote.fit_resample(input_feature_train_arr, target_feature_train_arr)
            
            train_arr = np.c_[input_feature_train_arr_balanced, target_feature_train_arr_balanced]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys)

