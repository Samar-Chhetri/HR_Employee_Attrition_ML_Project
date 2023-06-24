from flask import Flask, render_template, request

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attrition', methods = ['GET', 'POST'])
def predict_attrition():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            BusinessTravel = request.form.get('BusinessTravel'),
            DailyRate = int(request.form.get('DailyRate')),
            Department = request.form.get('Department'),
            DistanceFromHome = int(request.form.get('DistanceFromHome')),
            Education = int(request.form.get('Education')),
            EducationField = request.form.get('EducationField'),
            EnvironmentSatisfaction = int(request.form.get('EnvironmentSatisfaction')),
            Gender = request.form.get('Gender'),
            HourlyRate = int(request.form.get('HourlyRate')),
            JobInvolvement = int(request.form.get('JobInvolvement')),
            JobLevel = int(request.form.get('JobLevel')),
            JobRole = request.form.get('JobRole'),
            JobSatisfaction = int(request.form.get('JobSatisfaction')),
            MaritalStatus = request.form.get('MaritalStatus'),
            MonthlyIncome = int(request.form.get('MonthlyIncome')),
            MonthlyRate = int(request.form.get('MonthlyRate')),
            NumCompaniesWorked = int(request.form.get('NumCompaniesWorked')),
            OverTime = request.form.get('OverTime'),
            PercentSalaryHike = int(request.form.get('PercentSalaryHike')),
            PerformanceRating = int(request.form.get('PerformanceRating')),
            RelationshipSatisfaction = int(request.form.get('RelationshipSatisfaction')),
            StockOptionLevel = int(request.form.get('StockOptionLevel')),
            TotalWorkingYears = int(request.form.get('TotalWorkingYears')),
            TrainingTimesLastYear = int(request.form.get('TrainingTimesLastYear')),
            WorkLifeBalance = int(request.form.get('WorkLifeBalance')),
            YearsAtCompany = int(request.form.get('YearsAtCompany')),
            YearsInCurrentRole = int(request.form.get('YearsInCurrentRole')),
            YearsSinceLastPromotion = int(request.form.get('YearsSinceLastPromotion')),
            YearsWithCurrManager = int(request.form.get('YearsWithCurrManager'))

        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        results = results[0]
        if results == 0.0:
            results = 'The employee will stay in the company.'
        else:
            results = 'The employee will leave the company.'
            
        return render_template('home.html', results = results)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

