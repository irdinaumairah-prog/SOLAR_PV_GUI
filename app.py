import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solar PV Forecasting GUI",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS DESIGN
# ============================================================

st.markdown("""
<style>
.main-title {
    background: linear-gradient(90deg, #0f766e, #14b8a6);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.subtitle {
    font-size: 18px;
    color: #f0fdfa;
}

.card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

.small-text {
    color: #475569;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

col1, col2, col3 = st.columns([1,3,1])

with col2:
    import os

if os.path.exists("uthm.logo.jpg"):
    st.image("uthm.logo.jpg", width=130)
else:
    st.warning("UTHM logo not found. Please upload uthm.logo.jpg to GitHub.")

st.markdown("""
<div class="main-title">
    <h1>Deep Learning-Based Forecasting of Solar PV Output</h1>
    <p class="subtitle">for Industrrial Manufacturing Company</p>
    <p class="subtitle"><b>Prepared by:</b> Siti Irdina Umairah</p>
    <p class="subtitle"><b>Universiti Tun Hussein Onn Malaysia</b></p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_position" not in st.session_state:
    st.session_state.user_position = ""

# ============================================================
# FUNCTION: LOAD FORECAST DATA
# ============================================================

def load_forecast_data():
    file_name = "forecast_result.csv"

    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
    else:
        data = pd.DataFrame({
            "Month": [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            "Actual_2025": [
                4200, 4350, 4600, 4800, 5100, 5300,
                5500, 5400, 5200, 5000, 4700, 4500
            ],
            "Prediction_2026": [
                4300, 4450, 4700, 4950, 5200, 5450,
                5600, 5550, 5350, 5150, 4850, 4650
            ]
        })

    required_columns = ["Month", "Actual_2025", "Prediction_2026"]

    for col in required_columns:
        if col not in data.columns:
            st.error(f"Missing column in CSV file: {col}")
            st.stop()

    data["Actual_2025"] = pd.to_numeric(data["Actual_2025"], errors="coerce")
    data["Prediction_2026"] = pd.to_numeric(data["Prediction_2026"], errors="coerce")

    data = data.dropna(subset=["Actual_2025", "Prediction_2026"])

    return data

# ============================================================
# FUNCTION: SAVE USER LOGIN
# ============================================================

def save_login_record(name, email, position):
    new_record = pd.DataFrame({
        "Name": [name],
        "Email": [email],
        "Position": [position],
        "Login_Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    file_name = "user_login_records.csv"

    if os.path.exists(file_name):
        old_record = pd.read_csv(file_name)
        all_record = pd.concat([old_record, new_record], ignore_index=True)
    else:
        all_record = new_record

    all_record.to_csv(file_name, index=False)

# ============================================================
# LOGIN PAGE
# ============================================================

if st.session_state.logged_in == False:

    st.subheader("User Sign In")

    st.write("Please enter your details before accessing the forecasting dashboard.")

    with st.form("login_form"):

        name = st.text_input("Full Name")
        email = st.text_input("Email")
        position = st.selectbox(
            "Position",
            [
                "Student",
                "Supervisor",
                "Lecturer",
                "Engineer",
                "Technician",
                "Management",
                "Other"
            ]
        )

        submit = st.form_submit_button("Sign In")

        if submit:
            if name.strip() == "":
                st.warning("Please enter your full name.")
            elif email.strip() == "":
                st.warning("Please enter your email.")
            elif "@" not in email or "." not in email:
                st.warning("Please enter a valid email address.")
            else:
                save_login_record(name, email, position)

                st.session_state.logged_in = True
                st.session_state.user_name = name
                st.session_state.user_email = email
                st.session_state.user_position = position

                st.success("Sign in successful.")
                st.rerun()

# ============================================================
# MAIN GUI DASHBOARD
# ============================================================

else:

    df = load_forecast_data()

    st.sidebar.success(f"Logged in as: {st.session_state.user_name}")

    menu = st.sidebar.radio(
        "Navigation Menu",
        [
            "Dashboard",
            "Forecasting Result",
            "Performance Metrics",
            "User Login Records",
            "About Project"
        ]
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        st.session_state.user_position = ""
        st.rerun()

    # ========================================================
    # DASHBOARD PAGE
    # ========================================================

    if menu == "Dashboard":

        st.header("Dashboard Overview")

        total_month = len(df)
        average_actual = df["Actual_2025"].mean()
        average_prediction = df["Prediction_2026"].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Month", total_month)

        with col2:
            st.metric("Average Actual 2025", round(average_actual, 2))

        with col3:
            st.metric("Average Prediction 2026", round(average_prediction, 2))

        st.write("")

        st.subheader("Forecasting Dataset")
        st.dataframe(df, use_container_width=True)

        st.write("")

        st.info(
            "This dashboard shows the monthly average Solar PV output for actual data in 2025 "
            "and prediction data for 2026."
        )

    # ========================================================
    # FORECASTING RESULT PAGE
    # ========================================================

    elif menu == "Forecasting Result":

        st.header("Forecasting Result")

        st.write("This section displays the comparison between Actual 2025 and Prediction 2026.")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            df["Month"],
            df["Actual_2025"],
            marker="o",
            linewidth=2,
            label="Actual 2025"
        )

        ax.plot(
            df["Month"],
            df["Prediction_2026"],
            marker="o",
            linewidth=2,
            label="Prediction 2026"
        )

        ax.set_title("Monthly Average Solar PV Output: Actual 2025 vs Prediction 2026")
        ax.set_xlabel("Month")
        ax.set_ylabel("Solar PV Output")
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        st.subheader("Forecasting Result Table")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False)

        st.download_button(
            label="Download Forecasting Result CSV",
            data=csv_data,
            file_name="forecasting_result.csv",
            mime="text/csv"
        )

    # ========================================================
    # PERFORMANCE METRICS PAGE
    # ========================================================

    elif menu == "Performance Metrics":

        st.header("Model Performance Metrics")

        actual = df["Actual_2025"]
        prediction = df["Prediction_2026"]

        r2 = r2_score(actual, prediction)
        mae = mean_absolute_error(actual, prediction)
        rmse = math.sqrt(mean_squared_error(actual, prediction))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("R² Score", round(r2, 4))

        with col2:
            st.metric("MAE", round(mae, 4))

        with col3:
            st.metric("RMSE", round(rmse, 4))

        st.write("")

        metric_df = pd.DataFrame({
            "Metric": ["R² Score", "MAE", "RMSE"],
            "Value": [round(r2, 4), round(mae, 4), round(rmse, 4)]
        })

        st.subheader("Performance Metrics Table")
        st.dataframe(metric_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(metric_df["Metric"], metric_df["Value"])
        ax.set_title("Model Performance Metrics")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        plt.tight_layout()

        st.pyplot(fig)

        st.info(
            "R² Score shows how well the prediction follows the actual data pattern. "
            "MAE and RMSE show the forecasting error value. Lower MAE and RMSE indicate better model performance."
        )

    # ========================================================
    # USER LOGIN RECORDS PAGE
    # ========================================================

    elif menu == "User Login Records":

        st.header("User Login Records")

        if os.path.exists("user_login_records.csv"):
            login_df = pd.read_csv("user_login_records.csv")

            st.dataframe(login_df, use_container_width=True)

            st.download_button(
                label="Download User Login Records",
                data=login_df.to_csv(index=False),
                file_name="user_login_records.csv",
                mime="text/csv"
            )
        else:
            st.info("No user login records available yet.")

    # ========================================================
    # ABOUT PROJECT PAGE
    # ========================================================

    elif menu == "About Project":

        st.header("About Project")

        st.subheader("Project Title")
        st.success("Deep Learning-Based Forecasting of Solar PV Output for Smart Energy Management")

        st.subheader("Project Description")
        st.write("""
        This GUI system is developed to display the forecasting result of Solar PV output
        for an industrial manufacturing company. The system allows users to sign in using
        their name, email and position before accessing the forecasting dashboard.
        """)

        st.write("""
        The forecasting result is displayed using monthly average data. The system compares
        actual Solar PV output for 2025 with predicted Solar PV output for 2026. This helps
        users understand the expected solar energy generation pattern and supports smart
        energy management decisions.
        """)

        st.subheader("GUI Functions")
        st.write("""
        1. User sign in using name, email and position.  
        2. Display Solar PV forecasting dataset.  
        3. Display Actual 2025 vs Prediction 2026 graph.  
        4. Calculate and display R² Score, MAE and RMSE.  
        5. Save user login records automatically.  
        6. Allow user to download forecasting result and login records.
        """)

        st.subheader("System Development Tool")
        st.write("""
        This GUI is developed using Python and Streamlit. Streamlit is used to create
        a web-based graphical user interface that can be accessed through a browser.
        """)
