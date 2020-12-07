from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
import json
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

df = pd.read_csv(r'C:\Users\AIAAUser\Desktop\Purwadhika\JCDS\Final_Project\Dashboard\IBM_HR.csv')



df.Attrition.replace(('Yes', 'No'), (1, 0), inplace=True)
df.Gender.replace(('Male', 'Female'), (1, 0), inplace=True)
df.MaritalStatus.replace(('Divorced', 'Married', 'Single'), (0, 1, 2), inplace=True)
df.JobRole.replace(('Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                    'Manager', 'Manufacturing Director', 'Research Director',
                    'Research Scientist', 'Sales Executive', 'Sales Representative'),
                   (0, 1, 2, 3, 4, 5, 6, 7, 8), inplace=True)
df.OverTime.replace(('Yes', 'No'), (1, 0), inplace=True)

data = df[['Attrition' ,'Age', 'Gender', 'MaritalStatus',
           'TotalWorkingYears', 'MonthlyIncome', 'JobRole', 'OverTime']]

#Splitting Data into Train and Test sets
target = 'Attrition'
# data_dum = pd.get_dummies(data, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop([target], axis=1),
    data[target],
    random_state=101, test_size=0.3)

#SMOTE
sm = SMOTE(sampling_strategy="minority")
oversampled_trainX, oversampled_trainY = sm.fit_sample(X_train, y_train)

oversampled_trainX = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)


# home route
@app.route('/')
def home():
    return render_template('home.html')

#introduction route
@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

#dataset route
@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

#visualization route
@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

#conclusion route
@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        input = request.form

        Age = int(input['Age'])

        Gender = input['Gender']
        if Gender == 'Female':
            inputGender = 0
            strGender = 'Female'
        elif Gender == 'Male':
            inputGender = 1
            strGender = 'Male'

        MaritalStatus = input['MaritalStatus']
        if MaritalStatus == 'Single':
            inputMS = 2
            strMS = 'Single'
        elif MaritalStatus == 'Married':
            inputMS = 1
            strMS = 'Married'
        elif MaritalStatus == 'Divorced':
            inputMS = 0
            strMS = 'Divorced'

        TotalWorkingYears = int(input['TotalWorkingYears'])

        MonthlyIncome = int(input['MonthlyIncome'])

        JobRole = input['JobRole']
        if JobRole == 'Healthcare Representative':
            inputJR = 0
            strJR = 'Healthcare Representative'
        elif JobRole == 'Human Resources':
            inputJR = 1
            strJR = 'Human Resources'
        elif JobRole == 'Laboratory Technician':
            inputJR = 2
            strJR = 'Laboratory Technician'
        elif JobRole == 'Manager':
            inputJR = 3
            strJR = 'Manager'
        elif JobRole == 'Manufacturing Director':
            inputJR = 4
            strJR = 'Manufacturing Director'
        elif JobRole == 'Research Director':
            inputJR = 5
            strJR = 'Research Director'
        elif JobRole == 'Research Scientist':
            inputJR = 6
            strJR = 'Research Scientist'
        elif JobRole == 'Sales Executive':
            inputJR = 7
            strJR = 'Sales Executive'
        elif JobRole == 'Sales Representative':
            inputJR = 8
            strJR = 'Sales Representative'

        OverTime = input['OverTime']
        if OverTime == 'Yes':
            inputOT = 1
            strOT = 'Yes'
        elif OverTime == 'No':
            inputOT = 0
            strOT = 'No'


        input_predict = pd.DataFrame()

        input_predict['Age'] = [Age]
        input_predict['Gender'] = [inputGender]
        input_predict['MaritalStatus'] = [inputMS]
        input_predict['TotalWorkingYears'] = [TotalWorkingYears]
        input_predict['MonthlyIncome'] = [MonthlyIncome]
        input_predict['JobRole'] = [inputJR]
        input_predict['OverTime'] = [inputOT]



        predict = model.predict(input_predict)[0]

        if predict == 1:
            result = 'Yes_Attrition'
        elif predict == 0:
            result = 'No_Attriton'

        y_pred = model.predict(X_test)
        accu_score = round(recall_score(y_test, y_pred), 4)*100

        return render_template('result.html', Age = Age, Gender = strGender, MaritalStatus = strMS,
                               TotalWorkingYears = TotalWorkingYears, MonthlyIncome = MonthlyIncome, JobRole = strJR,
                               OverTime = strOT, accu_score = accu_score, result = result)

if __name__ == '__main__':
    rf_model = RandomForestClassifier(min_samples_leaf=6, min_samples_split=6,
                                      n_estimators=100)

    # fit the model
    model = rf_model.fit(oversampled_trainX.drop(['Attrition'], axis=1), oversampled_trainY)

    app.run(debug=True)

