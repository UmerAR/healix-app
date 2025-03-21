import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')
df = df.drop(['Course', 'CGPA', 'Social_Support', 'Substance_Use', 'Counseling_Service_Use', 'Family_History', 'Chronic_Illness', 'Semester_Credit_Load', 'Residence_Type'], axis="columns")

question_columns = ['Age', 'Gender', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 'Financial_Stress', 'Extracurricular_Involvement', 'Relationship_Status'] 
categorical_cols = ['Gender', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 'Extracurricular_Involvement', 'Relationship_Status']
target_columns  = ['Stress_Level', 'Depression_Score', 'Anxiety_Score']

label_encoders = {}  # Store encoders for future use
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert text to numbers
    label_encoders[col] = le  # Save encoder for later decoding

target_encoders = {}
for target in target_columns:
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    target_encoders[target] = le  # Save encoder for decoding predictions

models = {}
for target in target_columns:
    X = df[question_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    models[target] = model

# Save trained models and encoders
joblib.dump(models, "mental_health_models.pkl")
joblib.dump(target_encoders, "target_encoders.pkl")
joblib.dump(label_encoders, "feature_encoders.pkl")