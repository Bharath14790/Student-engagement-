import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="PragyanAI Dashboard", layout="wide")

# Title and Dataset Loading
st.title("PragyanAI - Student Engagement Intelligence System")
st.markdown("### LMS + Behavior → Learning → Placement")

try:
    df = pd.read_csv('student_data.csv')
    df['Placed'] = (df['Placement_Status'] == 'Placed').astype(int)
    
    # Sidebar for Navigation
    menu = st.sidebar.selectbox("Go to Section", ["Overview", "Engagement Analysis", "ML Insights"])

    if menu == "Overview":
        st.header("1. Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            place_rate = (df['Placed'].mean() * 100)
            st.metric("Placement Rate", f"{place_rate:.1f}%")
        with col3:
            avg_cgpa = df['CGPA'].mean()
            st.metric("Average CGPA", round(avg_cgpa, 2))

        # Chart: Placement by Dept
        fig1, ax1 = plt.subplots()
        dept_place = df.groupby('Department')['Placed'].mean() * 100
        dept_place.plot(kind='bar', color='#3b82f6', ax=ax1)
        ax1.set_title("Placement Rate by Department")
        st.pyplot(fig1)

    elif menu == "Engagement Analysis":
        st.header("2. Behavior vs Outcome")
        
        # Habits: Attendance and Logins
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Attendance vs Placement")
            fig_att, ax_att = plt.subplots()
            df['Att_Bucket'] = pd.cut(df['Attendance_%'], bins=[0,60,80,100], labels=['<60%','60-80%','>80%'])
            df.groupby('Att_Bucket')['Placed'].mean().plot(kind='bar', color='#22c55e', ax=ax_att)
            st.pyplot(fig_att)
            st.info("Insight: Consistency (>80% Attendance) is a high predictor of success.")

        with col_b:
            st.subheader("Doubt Behavior")
            fig_doubt, ax_doubt = plt.subplots()
            # Segmentation from your document [cite: 83, 90]
            df['Doubt_Cat'] = pd.cut(df['Doubts_Raised'], bins=[-1,0,4,100], labels=['No Doubts','Some Doubts','Active Doubts'])
            df.groupby('Doubt_Cat')['Placed'].mean().plot(kind='bar', color='#f59e0b', ax=ax_doubt)
            st.pyplot(fig_doubt)
            st.info("Insight: Asking doubts indicates a growth mindset.")

    elif menu == "ML Insights":
        st.header("3. Machine Learning & Segmentation")
        
        # Feature Importance
        feature_cols = ['Attendance_%', 'Login_Frequency', 'Time_Spent_Hours', 'Avg_Quiz_Score', 'Engagement_Score', 'CGPA']
        X = df[feature_cols]
        y = df['Placed']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
        fig_ml, ax_ml = plt.subplots()
        fi.plot(kind='barh', color='#0D1B3E', ax=ax_ml)
        ax_ml.set_title("Top Predictors of Placement")
        st.pyplot(fig_ml)
        
        # Student Segmentation [cite: 119]
        st.subheader("Student Segments")
        def get_segment(row):
            if row['Engagement_Score'] >= 65 and row['Avg_Quiz_Score'] >= 65: return 'High Performer'
            elif row['Engagement_Score'] < 40: return 'Disengaged'
            else: return 'Passive Learner'
        
        df['Segment'] = df.apply(get_segment, axis=1)
        st.table(df['Segment'].value_counts())

except FileNotFoundError:
    st.error("Error: 'student_data.csv' not found. Please ensure it is in your GitHub folder.")
