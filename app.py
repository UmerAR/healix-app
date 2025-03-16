from flask import Flask, render_template, request
import pandas as pd
import joblib
from scipy.stats import gmean
import google.generativeai as genai

API_KEY = ""

sys_instruct = "You are a medical proffesional, specialising in mental health "
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=sys_instruct)

app = Flask(__name__)

models = joblib.load("mental_health_models.pkl")
target_encoders = joblib.load("target_encoders.pkl")
label_encoders = joblib.load("feature_encoders.pkl")

question_columns = ['Age', 'Gender', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 'Financial_Stress', 'Extracurricular_Involvement', 'Relationship_Status'] 
categorical_cols = ['Gender', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 'Extracurricular_Involvement', 'Relationship_Status']
target_columns  = ['Stress_Level', 'Depression_Score', 'Anxiety_Score']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():
    response = None
    user_prompt = None
    if request.method == 'POST':
        try:
            user_prompt = request.form["message"]
            
            prompt = (
                f'You are a compassionate mental health assistant. ',
                f"A user said: \"{user_prompt}\"",
                f'Respond in a conversational manner, offering empathetic support, active listening, and gentle advice to help improve their mental wellbeing. '
                f"Maintain a friendly tone and help the user out with their problems."
            )
            response = model.generate_content(prompt).text

        except Exception as e:
            print(f'error: {e}')

    return render_template(
        'chatbot.html',
        response=response,
        user_prompt=user_prompt
    )

@app.route('/macros', methods=["GET", "POST"])
def macros():
    bmi = None
    category = None
    calories = None
    if request.method == 'POST':
        try:
            weight = float(request.form["weight"])
            heightCM = float(request.form["height"])
            heightM = float(request.form["height"]) / 100  # Convert cm to meters
            age = int(request.form["age"])
            activityMultiplier = float(request.form["activity"])
            gender = request.form["gender"]

            bmi = round(weight / (heightM ** 2), 2)

            if bmi < 18.5:
                category = "Underweight"
            elif 18.5 <= bmi < 24.9:
                category = "Normal weight"
            elif 25 <= bmi < 29.9:
                category = "Overweight"
            else:
                category = "Obese"

            if gender == "Male":
                calories = (10*weight) + (6.25*heightCM) - (5*age) + 5
            elif gender == "Female":
                calories = (10*weight) + (6.25*heightCM) - (5*age) - 161

            calories = round(calories * activityMultiplier, 2)

        except Exception as e:
            print(f'error: {e}')
    
    return render_template(
        'macro.html',
        bmi=bmi,
        category=category,
        calories=calories
    )

@app.route('/stress', methods=["GET", "POST"])
def stress():
    stress_score = None
    predictions = {"Stress_Level": "N/A", "Depression_Score": "N/A", "Anxiety_Score": "N/A"}

    if request.method == 'POST':
        try:
            user_data = {}
            for question in question_columns:
                value = request.form.get(question.lower())  # Match form field names
                if question in categorical_cols:
                    user_data[question] = label_encoders[question].transform([value])[0]
                else:
                    user_data[question] = int(value)

            user_df = pd.DataFrame([user_data])

            for target in models:
                model = models[target]
                pred = model.predict(user_df)[0]
                predictions[target] = target_encoders[target].inverse_transform([pred])[0]

            gmean_value = gmean([target_encoders[target].transform([predictions[target]])[0] + 1 for target in models]) - 1
            stress_score = round(gmean_value * 20, 2)

        except Exception as e:
            print(f'Error: {e}')
            predictions = {"Stress_Level": "N/A", "Depression_Score": "N/A", "Anxiety_Score": "N/A"}

    return render_template(
        'stress.html',
        stress_score=stress_score,
        stress_level=predictions['Stress_Level'],
        anxiety_score=predictions["Anxiety_Score"],
        depression_score=predictions["Depression_Score"]                       
    )

if __name__ == "__main__":
    app.run(debug=True)
