import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title="PragyanAI Student Portal", layout="wide")

# Your exact file name
file_name = 'student_data.csv'

@st.cache_data
def load_data():
    try:
        # Loading your CSV file
        return pd.read_csv(file_name)
    except Exception as e:
        return None

df = load_data()

if df is not None:
    st.title("PRAGYAN ΔΙ - Intelligence System")
    st.markdown("#### Grow with Gyan: Student Engagement Portal")

    # Sidebar Search
    st.sidebar.header("Student Search")
    search_id = st.sidebar.text_input("Enter Student ID / USN")

    if search_id:
        # Filter for the specific student [cite: 9]
        student_row = df[df['Student_ID'] == search_id]
        
        if not student_row.empty:
            s = student_row.iloc[0]
            st.success(f"Displaying Profile for: {search_id}")
            
            # Header Profile details [cite: 10, 12, 13, 56]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Department", s['Department'])
            c2.metric("College", s['College'])
            c3.metric("CGPA", s['CGPA'])
            c4.metric("Placement Status", s['Placement_Status'])

            st.divider()

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("📊 Platform Activity")
                # Core LMS Logs 
                metrics_df = pd.DataFrame({
                    "Activity": ["Attendance %", "Login Frequency", "Time Spent (Hrs)", "Active Days"],
                    "Status": [f"{s['Attendance_%']}%", s['Login_Frequency'], s['Time_Spent_Hours'], s['Active_Days']]
                })
                st.table(metrics_df)

            with col_right:
                st.subheader("📚 Learning Progress")
                # Learning Engagement [cite: 30, 36, 40, 47]
                learning_df = pd.DataFrame({
                    "Metric": ["Video Completion %", "Quiz Score", "Doubts Raised", "Hackathons"],
                    "Value": [f"{s['Video_Completion_%']}%", s['Avg_Quiz_Score'], s['Doubts_Raised'], s['Hackathons']]
                })
                st.table(learning_df)

            # Intelligence Engine Scores [cite: 109, 115]
            st.divider()
            st.subheader("🧠 Placement Readiness")
            
            # Use the Engagement Score from your file
            score = int(s['Engagement_Score'])
            st.progress(score)
            st.write(f"The Engagement Intelligence Score is **{score}/100**")
            
            # Show Insight based on your project goals [cite: 66, 164]
            if score > 80:
                st.info("Insight: High consistency detected. This student is placement-ready.")
        else:
            st.error("Student ID not found in the database.")
    else:
        st.info("Please enter a Student ID in the sidebar to begin.")
else:
    st.error(f"File '{file_name}' not found. Ensure it is uploaded to GitHub and named correctly.")
