import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="PragyanAI Student Intelligence", layout="wide")

# 1. Load the Excel Dataset
file_name = 'PragyanAI_Student_Data_50k.xlsx'

try:
    # We read the 'Student_Data' sheet specifically
    df = pd.read_csv('student_data.csv') if 'csv' in file_name else pd.read_excel(file_name, sheet_name='Student_Data')
    
    st.title("PRAGYAN ΔΙ - Intelligence System")
    st.markdown("### Student Engagement & Placement Analytics")

    # 2. Sidebar Search
    st.sidebar.header("Student Search")
    search_id = st.sidebar.text_input("Enter Student USN (e.g., 2026USN10001)")

    if search_id:
        # Filter for the specific student
        student_row = df[df['Student_ID'] == search_id]
        
        if not student_row.empty:
            s = student_row.iloc[0]
            
            # Header Profile
            st.success(f"Displaying Profile for: {search_id}") [cite: 9, 59]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Department", s['Department']) [cite: 12]
            c2.metric("College Tier", s['College_Tier']) [cite: 11]
            c3.metric("CGPA", s['CGPA']) [cite: 13]
            c4.metric("Placement Status", s['Placement_Status']) [cite: 56]

            st.divider()

            # Engagement Metrics from Project Formula
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Platform Activity") [cite: 23, 140]
                metrics_df = pd.DataFrame({
                    "Activity": ["Attendance %", "Login Frequency", "Time Spent (Hrs)", "Active Days"], [cite: 17, 24, 25, 26]
                    "Status": [f"{s['Attendance_%']}%", s['Login_Frequency'], s['Time_Spent_Hours'], s['Active_Days']]
                })
                st.table(metrics_df)

            with col_right:
                st.subheader("📚 Learning Progress") [cite: 28, 144]
                learning_df = pd.DataFrame({
                    "Metric": ["Video Completion %", "Quiz Score", "Doubts Raised", "Hackathons"], [cite: 30, 36, 40, 47]
                    "Value": [f"{s['Video_Completion_%']}%", s['Quiz_Score'], s['Doubts_Raised'], s['Hackathons']]
                })
                st.table(learning_df)

            # Advanced AI Score
            st.divider()
            st.subheader("🧠 Intelligence Engine Score") [cite: 170]
            st.progress(int(s['Engagement_Score'])) [cite: 108]
            st.write(f"The Engagement Intelligence Score is **{s['Engagement_Score']}/100**") [cite: 109]

        else:
            st.error("Student ID not found in the 50k database. Please check the USN.")
    else:
        st.info("Please enter a Student USN in the sidebar to view individual engagement details.")

# 3. Error Handling if file is missing
except Exception as e:
    st.error(f"CRITICAL ERROR: File '{file_name}' not found.")
    st.warning("Please ensure the 50,000 row Excel file is uploaded to the same folder as this script.")
