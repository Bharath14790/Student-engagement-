import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Student Engagement Portal", layout="wide")

# Your exact file name
file_name = 'student_data.csv'

@st.cache_data
def load_data():
    try:
        return pd.read_csv(file_name)
    except Exception as e:
        return None

df = load_data()

if df is not None:
    st.title("Student Intelligence System")
    st.markdown("### Student Performance Dashboard")

    # Sidebar Search
    st.sidebar.header("Search Interface")
    search_id = st.sidebar.text_input("Enter Student ID / USN")

    if search_id:
        student_row = df[df['Student_ID'] == search_id]
        
        if not student_row.empty:
            s = student_row.iloc[0]
            st.success(f"Profile Loaded: {search_id}")
            
            # Key Academic Indicators
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Department", s['Department'])
            c2.metric("College", s['College'])
            c3.metric("CGPA", s['CGPA'])
            c4.metric("Status", s['Placement_Status'])

            st.divider()

            # Engagement and Learning Tables
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Platform Activity")
                activity_df = pd.DataFrame({
                    "Activity": ["Attendance %", "Login Frequency", "Time Spent (Hrs)", "Active Days"],
                    "Value": [f"{s['Attendance_%']}%", s['Login_Frequency'], s['Time_Spent_Hours'], s['Active_Days']]
                })
                st.table(activity_df)

            with col_right:
                st.subheader("📚 Learning Progress")
                learning_df = pd.DataFrame({
                    "Metric": ["Video Completion %", "Quiz Score", "Doubts Raised", "Hackathons"],
                    "Value": [
                        f"{s['Video_Completion_%']}%", 
                        s['Quiz_Score'], 
                        s['Doubts_Raised'], 
                        s['Hackathons']
                    ]
                })
                st.table(learning_df)

            # Analysis Score
            st.divider()
            st.subheader("🧠 Overall Engagement Analysis")
            
            score = int(s['Engagement_Score'])
            st.progress(score)
            st.write(f"The calculated Engagement Score is **{score}/100**")
            
            if score > 80:
                st.info("Analysis: This student shows high consistency across all metrics.")
        else:
            st.error("Student ID not found in the database.")
    else:
        st.info("Please enter a Student ID in the sidebar to display data.")
else:
    st.error(f"Data file '{file_name}' not found. Please ensure it is uploaded correctly.")
