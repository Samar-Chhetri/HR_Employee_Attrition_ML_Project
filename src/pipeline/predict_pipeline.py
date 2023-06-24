import os, sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,BusinessTravel:str, DailyRate:int, Department:str,DistanceFromHome:int, Education:int, EducationField:str, 
                 EnvironmentSatisfaction:int, Gender:str, HourlyRate:int,JobInvolvement:int, JobLevel:int, JobRole:str, JobSatisfaction:int,
                 MaritalStatus:str, MonthlyIncome:int, MonthlyRate:int, NumCompaniesWorked:int, OverTime:str, PercentSalaryHike:int, 
                 PerformanceRating:int,RelationshipSatisfaction:int, StockOptionLevel:int,TotalWorkingYears:int, TrainingTimesLastYear:int, 
                 WorkLifeBalance:int,YearsAtCompany:int, YearsInCurrentRole:int, YearsSinceLastPromotion:int,YearsWithCurrManager:int):
        
        self.BusinessTravel = BusinessTravel
        self.DailyRate = DailyRate
        self.Department = Department
        self.DistanceFromHome = DistanceFromHome
        self.Education = Education
        self.EducationField = EducationField
        self.EnvironmentSatisfaction = EnvironmentSatisfaction
        self.Gender = Gender
        self.HourlyRate = HourlyRate
        self.JobInvolvement = JobInvolvement
        self.JobLevel = JobLevel
        self.JobRole = JobRole
        self.JobSatisfaction = JobSatisfaction
        self.MaritalStatus = MaritalStatus
        self.MonthlyIncome = MonthlyIncome
        self.MonthlyRate = MonthlyRate
        self.NumCompaniesWorked = NumCompaniesWorked
        self.OverTime = OverTime
        self.PercentSalaryHike = PercentSalaryHike
        self.PerformanceRating = PerformanceRating
        self.RelationshipSatisfaction = RelationshipSatisfaction
        self.StockOptionLevel = StockOptionLevel
        self.TotalWorkingYears = TotalWorkingYears
        self.TrainingTimesLastYear = TrainingTimesLastYear
        self.WorkLifeBalance = WorkLifeBalance
        self.YearsAtCompany = YearsAtCompany
        self.YearsInCurrentRole = YearsInCurrentRole
        self.YearsSinceLastPromotion = YearsSinceLastPromotion
        self.YearsWithCurrManager = YearsWithCurrManager

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "BusinessTravel": [self.BusinessTravel],
                "DailyRate": [self.DailyRate],
                "Department": [self.Department],
                "DistanceFromHome": [self.DistanceFromHome],
                "Education": [self.Education],
                "EducationField": [self.EducationField],
                "EnvironmentSatisfaction": [self.EnvironmentSatisfaction],
                "Gender": [self.Gender],
                "HourlyRate": [self.HourlyRate],
                "JobInvolvement": [self.JobInvolvement],
                "JobLevel": [self.JobLevel],
                "JobRole": [self.JobRole],
                "JobSatisfaction": [self.JobSatisfaction],
                "MaritalStatus": [self.MaritalStatus],
                "MonthlyIncome": [self.MonthlyIncome],
                "MonthlyRate": [self.MonthlyRate],
                "NumCompaniesWorked": [self.NumCompaniesWorked],
                "OverTime": [self.OverTime],
                "PercentSalaryHike": [self.PercentSalaryHike],
                "PerformanceRating": [self.PerformanceRating],
                "RelationshipSatisfaction": [self.RelationshipSatisfaction],
                "StockOptionLevel": [self.StockOptionLevel],
                "TotalWorkingYears": [self.TotalWorkingYears],
                "TrainingTimesLastYear": [self.TrainingTimesLastYear],
                "WorkLifeBalance": [self.WorkLifeBalance],
                "YearsAtCompany": [self.YearsAtCompany],
                "YearsInCurrentRole": [self.YearsInCurrentRole],
                "YearsSinceLastPromotion": [self.YearsSinceLastPromotion],
                "YearsWithCurrManager": [self.YearsWithCurrManager]
            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)
        
        