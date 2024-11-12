# Student-Performance-Prediction

This repository contains the implementation of machine learning models for predicting student performance based on academic and non-academic features. The project aims to support early interventions for students at risk of underperforming academically. A Streamlit web application also allows users to predict student performance based on input features.

## Project Overview

### 1. Academic Performance Prediction

A machine learning model was trained to predict whether a student is a "Pass" or "Fail" based on the Exam Score. This model achieved an accuracy of 90%, indicating strong prediction capabilities based on the provided dataset.

### 3. Streamlit Application

A user-friendly web application was created using Streamlit, enabling users to input specific student data (attendance, hours studied, previous score, access to resources, tutoring sessions) and receive predictions on the student's "Grade". This application can be used by educators to support decision-making.

## How to Install the Streamlit Application

#### 1. Install the required libraries:

```
pip install -r requirements.txt
```

#### 2. Clone this repository

```
git clone 
```

#### 3. Navigate to the project directory in your machine

```
cd C:\Users\Documents\Student-Performance-Prediction
```

#### 4. Run the Streamlit app

```
streamlit run home.py
```

#### 5. A browser window will open with the application interface, where you can input student data to get the "Grade" of the student.

## Dataset

The dataset used for training our model was sourced from Kaggle's Student Performance Dataset [Link Text]( https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data ), which includes various features such as attendance, study hours, previous scores, and motivation level etc. These factors contribute to predicting student performance.

## Model Accuracy
