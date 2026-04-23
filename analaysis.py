import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Student Portal", layout="wide")

# 1. Load Dataset
file_name = 'student_data.csv'

try:
    # Reading the Student Data sheet from your Excel file
    df = pd.read_excel(file_name, sheet_name='Student_Data')
    
    st.title("PRAGYAN ΔΙ - Intelligence System")
    st.markdown("#### Student Engagement & Placement Analytics")

    # 2. Sidebar Search
    st.sidebar.header("Student Search")
    search_id = st.sidebar.text_input("Enter Student USN (e.g., 2026USN10001)")

    if search_id:
        student_row = df[df['Student_ID'] == search_id]
        
        if not student_row.empty:
            s = student_row.iloc[0]
            
            # Header Profile - Using your requested fields
            st.success(f"Displaying Profile for: {search_id}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Department", s['Department'])
            c2.metric("College Tier", s['College_Tier'])
            c3.metric("CGPA", s['CGPA'])
            c4.metric("Placement Status", s['Placement_Status'])

            st.divider()

            # Engagement & Activity Analysis
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Platform Activity")
                # Removed the citation that caused the SyntaxError
                metrics_df = pd.DataFrame({
                    "Activity": ["Attendance %", "Login Frequency", "Time Spent (Hrs)", "Active Days"],
                    "Status": [f"{s['Attendance_%']}%", s['Login_Frequency'], s['Time_Spent_Hours'], s['Active_Days']]
                })
                st.table(metrics_df)

            with col_right:
                st.subheader("📚 Learning Progress")
                learning_df = pd.DataFrame({
                    "Metric": ["Video Completion %", "Quiz Score", "Doubts Raised", "Hackathons"],
                    "Value": [f"{s['Video_Completion_%']}%", s['Quiz_Score'], s['Doubts_Raised'], s['Hackathons']]
                })
                st.table(learning_df)

            # Advanced AI Score
            st.divider()
            st.subheader("🧠 Intelligence Engine Score")
            st.progress(int(s['Engagement_Score']))
            st.write(f"The Engagement Intelligence Score is **{s['Engagement_Score']}/100**")

        else:
            st.error("Student ID not found in the 50k database.")
    else:
        st.info("Please enter a Student USN in the sidebar.")

except Exception as e:
    st.error(f"File '{file_name}' not found. Please upload it to GitHub.")
