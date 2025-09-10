import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Database setup
# -----------------------------
conn = sqlite3.connect('student_data.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    attendance INTEGER,
    quiz_score INTEGER,
    homework_score INTEGER,
    passed INTEGER
)
''')
conn.commit()

# -----------------------------
# Streamlit Website Layout
# -----------------------------
st.title("EduPredict: Student Performance Predictor")
st.write("Predict future academic performance using machine learning!")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload student CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Ensure required numeric columns exist
        required_cols = ['attendance', 'quiz_score', 'homework_score', 'passed']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0  # If missing, fill with 0

        # Convert numeric columns to numbers, fill invalid/missing values with 0
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure 'name' column exists
        if 'name' not in df.columns:
            df['name'] = [f"Student_{i+1}" for i in range(len(df))]
        else:
            df['name'] = df['name'].fillna([f"Student_{i+1}" for i in range(len(df))])

        st.write("### Uploaded Student Data")
        st.dataframe(df)

        # -----------------------------
        # ML Model
        # -----------------------------
        X = df[['attendance', 'quiz_score', 'homework_score']]
        y = df['passed']

        if len(df) < 2:
            st.warning("Not enough data for prediction. Add more rows.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            st.write(f"**Model Accuracy:** {acc*100:.2f}%")

            # Predict all students
            df['prediction'] = model.predict(X)
            df['risk'] = df['prediction'].apply(lambda x: 'Low' if x == 1 else 'High')

            st.write("### Predictions & Risk Scores")
            st.dataframe(df)

            # -----------------------------
            # Save to database
            # -----------------------------
            for index, row in df.iterrows():
                cursor.execute('''
                INSERT INTO students (name, attendance, quiz_score, homework_score, passed)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    row['name'],
                    row['attendance'], 
                    row['quiz_score'], 
                    row['homework_score'], 
                    row['prediction']
                ))
            conn.commit()

            # -----------------------------
            # Data Visualization
            # -----------------------------
            st.write("### Attendance vs Homework Scores")
            plt.figure(figsize=(8,5))
            plt.scatter(df['attendance'], df['homework_score'], c=df['prediction'], cmap='bwr', alpha=0.7)
            plt.xlabel("Attendance")
            plt.ylabel("Homework Score")
            plt.colorbar(label='Prediction (0=Fail, 1=Pass)')
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

st.write("✅ Database updated: student_data.db")
