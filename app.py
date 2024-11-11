import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from PIL import Image


# Apply custom CSS (Optional, adjust "style.css" if you want custom styling)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("StudentPerformanceFactors.csv")
    return data

data = load_data()

# Add a 'Grade' column based on the Exam_Score threshold
data['Grade'] = data['Exam_Score'].apply(lambda x: 'Pass' if x >= 65 else 'Fail')

# Encode categorical variables using LabelEncoder (like Access_to_Resources)
label_encoder = LabelEncoder()
data['Access_to_Resources'] = label_encoder.fit_transform(data['Access_to_Resources'])


# Sidebar for navigation
st.sidebar.title("Student Performance Prediction")
page = st.sidebar.selectbox("Select a page", ["Home", "Data Visualization", "Prediction"])

# Home page
if page == "Home":
    st.title("Welcome to the Student Performance Prediction App")
    st.image('im8.gif', use_column_width=True)
    st.write(
        """
        This app predicts student performance based on various factors. 
        Use the Data Visualization page to explore insights in the dataset,
        or head to the Prediction page to make predictions based on your inputs.
        """
    )

# Data Visualization page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("Select features to visualize their relationships with student performance.")
    
    if st.checkbox("Show Correlation Heatmap"):
        st.write("Correlation Heatmap of Features")
        numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
        
        if not numeric_data.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
            plt.clf()

    chart_type = st.selectbox("Select chart type", ["Histogram", "Scatter", "Box"])

    if chart_type == "Histogram":
        feature = st.selectbox("Select feature for histogram", data.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True, color='blue')
        st.pyplot(plt)
    else:
        x_var = st.selectbox("Select X-axis variable", data.columns)
        y_var = st.selectbox("Select Y-axis variable", data.columns)
        
        plt.figure(figsize=(10, 6))
        if chart_type == "Scatter":
            sns.scatterplot(x=data[x_var], y=data[y_var])
        elif chart_type == "Box":
            sns.boxplot(x=x_var, y=y_var, data=data)
        st.pyplot(plt)

# Prediction page using Naive Bayes
elif page == "Prediction":
    st.title("Performance Prediction")

    # Feature selection based on user requirements
    feature_cols = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Access_to_Resources', 'Tutoring_Sessions']
    X = data[feature_cols]
    y = data['Grade']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Naive Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Test accuracy display
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # User input fields
    st.write("Enter the student's details for prediction:")
    attendance = st.number_input("Attendance", min_value=0, max_value=100, step=1)
    hours_studied = st.number_input("Hours Studied", min_value=0, max_value=100, step=1)
    previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, step=1)
    access_to_resources = st.selectbox("Access to Resources", ["Yes", "No"])
    tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=100, step=1)

    # Encoding the categorical input for "Access to Resources"
    access_to_resources_encoded = 1 if access_to_resources == "Yes" else 0

    if st.button("Predict Performance"):
        # Prediction with user input
        new_data = [[attendance, hours_studied, previous_scores, access_to_resources_encoded, tutoring_sessions]]
        prediction = model.predict(new_data)

        # Show pass/fail GIF based on prediction
        if prediction[0] == 'Pass':
            st.image("im6.gif", caption="Congratulations! The student is predicted to pass.", use_column_width=True)
        else:
            st.image("im2.gif", caption="The student is predicted to fail. Consider additional support.", use_column_width=True)

        # Show entered details and prediction result
        st.write("**Entered Details:**")
        st.write(f"Attendance: {attendance}%")
        st.write(f"Hours Studied: {hours_studied}")
        st.write(f"Previous Scores: {previous_scores}%")
        st.write(f"Access to Resources: {access_to_resources}")
        st.write(f"Tutoring Sessions: {tutoring_sessions}")
        st.write(f"**Prediction Result:** {prediction[0]}")

        # Highlight Grade with color
        if prediction[0] == 'Pass':
            st.markdown(f"<h3 style='color: green;'>Predicted Grade: {prediction[0]}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red;'>Predicted Grade: {prediction[0]}</h3>", unsafe_allow_html=True)

        
