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
    page_title="Solar PV Forecasting System",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# CUSTOM CSS DESIGN
# ============================================================

st.markdown("""
/* About Project text styling */
{
  
}
<style>

/* Main app background */
.stApp {
    background-color: #f4f7fb;
}

/* Header card */
.main-title {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #ca8a04);
    padding: 38px;
    border-radius: 24px;
    color: white;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.25);
}

.main-title h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 23px;
    color: #fef3c7;
    font-weight: 500;
    margin-bottom: 6px;
}

.prepared {
    font-size: 19px;
    color: #ffffff;
    font-weight: 500;
    margin-bottom: 4px;
}

/* Logo container */
.logo-card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 22px;
    margin-bottom: 18px;
    box-shadow: 0px 4px 14px rgba(15, 23, 42, 0.10);
    text-align: center;
}

/* General section card */
.section-card {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #dbeafe;
    box-shadow: 0px 6px 18px rgba(15, 23, 42, 0.08);
    margin-bottom: 24px;
}

/* Login page */
.login-card {
    background: linear-gradient(135deg, #eff6ff, #ffffff);
    padding: 34px;
    border-radius: 24px;
    border: 2px solid #bfdbfe;
    box-shadow: 0px 8px 22px rgba(37, 99, 235, 0.12);
    margin-bottom: 24px;
}

.login-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
}

.login-subtitle {
    text-align: center;
    font-size: 17px;
    color: #475569;
    margin-bottom: 20px;
}

/* Streamlit input labels */
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    color: #0f172a !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* Text input boxes */
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 2px solid #2563eb !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* Placeholder text */
div[data-testid="stTextInput"] input::placeholder {
    color: #64748b !important;
    font-size: 15px !important;
}

/* Selectbox - Position box */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 2px solid #2563eb !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    font-size: 16px !important;
}

/* Selectbox selected text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #0f172a !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Selectbox placeholder text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #0f172a !important;
    opacity: 1 !important;
}

/* Selectbox dropdown arrow */
div[data-testid="stSelectbox"] svg {
    color: #2563eb !important;
    fill: #2563eb !important;
}

/* Form submit button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #1e3a8a, #2563eb) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    padding: 11px 28px !important;
    border: none !important;
}

/* Info boxes */
.info-box {
    background-color: #eff6ff;
    padding: 20px;
    border-radius: 16px;
    border-left: 7px solid #2563eb;
    color: #1e3a8a;
    font-size: 16px;
    line-height: 1.6;
}

.success-box {
    background-color: #ecfdf5;
    padding: 20px;
    border-radius: 16px;
    border-left: 7px solid #059669;
    color: #064e3b;
    font-size: 16px;
    line-height: 1.6;
}

.warning-box {
    background-color: #fffbeb;
    padding: 20px;
    border-radius: 16px;
    border-left: 7px solid #ca8a04;
    color: #713f12;
    font-size: 16px;
    line-height: 1.6;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #cbd5e1;
}
/* Global text color for main content */
section.main,
section.main p,
section.main div,
section.main span,
section.main label,
section.main h1,
section.main h2,
section.main h3,
section.main h4,
section.main h5,
section.main h6 {
    color: #0f172a !important;
}

/* Streamlit markdown text */
div[data-testid="stMarkdownContainer"] {
    color: #0f172a !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

/* Dataframe and table text */
div[data-testid="stDataFrame"] {
    color: #0f172a !important;
}

/* Metric label and value */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 18px !important;
    border-radius: 16px !important;
    border: 1px solid #dbeafe !important;
    box-shadow: 0px 4px 12px rgba(15, 23, 42, 0.08) !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: #0f172a !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] {
    background-color: #eaf2ff !important;
}

section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* Info, success and warning text */
.info-box,
.info-box *,
.success-box,
.success-box *,
.warning-box,
.warning-box * {
    color: inherit !important;
}
/* Keep main title text white/gold */
.main-title,
.main-title h1,
.main-title p,
.main-title b {
    color: white !important;
}

.main-title .subtitle {
    color: #fef3c7 !important;
}

.main-title .prepared {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER WITH OPTIONAL LOGO
# ============================================================

st.markdown('<div class="logo-card">', unsafe_allow_html=True)

if os.path.exists("uthm.logo.jpg.new"):
    st.image("uthm.logo.jpg.new", width=360)
else:
    st.markdown(
        "<h3 style='color:#1e3a8a;'>Universiti Tun Hussein Onn Malaysia</h3>",
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>Deep Learning-Based Forecasting of Solar PV Output</h1>
    <p class="subtitle">for Industrial Manufacturing Company</p>
    <p class="prepared"><b>Prepared by:</b> Siti Irdina Umairah</p>
    <p class="prepared"><b>Universiti Tun Hussein Onn Malaysia</b></p>
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

    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown('<p class="login-title">User Sign In</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="login-subtitle">Please fill in your name, email address and position to access the forecasting dashboard.</p>',
        unsafe_allow_html=True
    )

    with st.form("login_form"):

        name = st.text_input(
            "Full Name",
            placeholder="Example: Siti Irdina Umairah"
        )

        email = st.text_input(
            "Email Address",
            placeholder="Example: sitiirdina@gmail.com"
        )

        position = st.selectbox(
    "Position",
    [
        "Select your position",
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
                st.warning("Please enter your email address.")
            elif "@" not in email or "." not in email:
                st.warning("Please enter a valid email address.")
            elif position == "Please select your position":
                st.warning("Please select your position.")
            else:
                save_login_record(name, email, position)

                st.session_state.logged_in = True
                st.session_state.user_name = name
                st.session_state.user_email = email
                st.session_state.user_position = position

                st.success("Sign in successful.")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>System Information:</b><br>
        This GUI system allows users to view Solar PV forecasting results, performance metrics,
        and smart energy management insights through an interactive web-based dashboard.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN GUI DASHBOARD
# ============================================================

else:

    df = load_forecast_data()

    # Sidebar user profile
    st.sidebar.success("Login Successful")
    st.sidebar.write(f"👤 **Name:** {st.session_state.user_name}")
    st.sidebar.write(f"📧 **Email:** {st.session_state.user_email}")
    st.sidebar.write(f"💼 **Position:** {st.session_state.user_position}")
    st.sidebar.write("---")

    menu = st.sidebar.radio(
        "Navigation Menu",
        [
            "Dashboard",
            "Forecasting Result",
            "Performance Metrics",
            "Energy Management Insight",
            "User Login Records",
            "About Project"
        ]
    )

    st.sidebar.write("---")

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
        total_prediction = df["Prediction_2026"].sum()
        highest_month = df.loc[df["Prediction_2026"].idxmax(), "Month"]
        highest_value = df["Prediction_2026"].max()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Month", total_month)

        with col2:
            st.metric("Average Actual 2025", round(average_actual, 2))

        with col3:
            st.metric("Average Prediction 2026", round(average_prediction, 2))

        with col4:
            st.metric("Highest Forecast Month", highest_month)

        st.write("")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""
            <div class="success-box">
                <b>System Status:</b><br>
                Forecasting dashboard is active and ready to display Solar PV prediction results.
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
            <div class="warning-box">
                <b>Highest Predicted Output:</b><br>
                {highest_month} shows the highest predicted Solar PV output with a value of {round(highest_value, 2)}.
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.subheader("Forecasting Dataset")
        st.dataframe(df, use_container_width=True)

        st.write("")

        st.markdown("""
        <div class="info-box">
            The dashboard displays monthly average Solar PV output based on actual data for 2025
            and predicted data for 2026. This helps users understand the solar energy generation
            trend and supports smart energy management decisions.
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # FORECASTING RESULT PAGE
    # ========================================================

    elif menu == "Forecasting Result":

        st.header("Forecasting Result")

        st.markdown("""
        <div class="info-box">
            This section shows the comparison between Actual 2025 and Prediction 2026 Solar PV output.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.subheader("Line Graph: Actual 2025 vs Prediction 2026")

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

        st.write("")

        st.subheader("Bar Chart: Monthly Comparison")

        fig2, ax2 = plt.subplots(figsize=(12, 6))

        x = range(len(df["Month"]))
        width = 0.35

        ax2.bar([i - width/2 for i in x], df["Actual_2025"], width, label="Actual 2025")
        ax2.bar([i + width/2 for i in x], df["Prediction_2026"], width, label="Prediction 2026")

        ax2.set_title("Monthly Comparison: Actual 2025 vs Prediction 2026")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Solar PV Output")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["Month"], rotation=45)
        ax2.legend()
        ax2.grid(axis="y")

        plt.tight_layout()
        st.pyplot(fig2)

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

        st.markdown("""
        <div class="info-box">
            <b>Explanation:</b><br>
            R² Score shows how well the prediction follows the actual data pattern.
            MAE and RMSE show the forecasting error value. Lower MAE and RMSE indicate
            better model performance.
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # ENERGY MANAGEMENT INSIGHT PAGE
    # ========================================================

    elif menu == "Energy Management Insight":

        st.header("Energy Management Insight")

        highest_month = df.loc[df["Prediction_2026"].idxmax(), "Month"]
        highest_value = df["Prediction_2026"].max()
        lowest_month = df.loc[df["Prediction_2026"].idxmin(), "Month"]
        lowest_value = df["Prediction_2026"].min()
        average_prediction = df["Prediction_2026"].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Highest Forecast Month", highest_month)

        with col2:
            st.metric("Lowest Forecast Month", lowest_month)

        with col3:
            st.metric("Average Prediction", round(average_prediction, 2))

        st.write("")

        st.subheader("Smart Energy Management Recommendations")

        st.markdown(f"""
        <div class="success-box">
            <b>1. High Solar Generation Period</b><br>
            The highest predicted Solar PV output occurs in <b>{highest_month}</b>.
            During this period, the industrial manufacturing company can schedule high-energy
            operations when solar generation is expected to be higher.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown(f"""
        <div class="warning-box">
            <b>2. Low Solar Generation Period</b><br>
            The lowest predicted Solar PV output occurs in <b>{lowest_month}</b>.
            During this period, the company should manage energy usage carefully and may need
            support from grid electricity or backup energy sources.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <div class="info-box">
            <b>3. Energy Optimization Strategy</b><br>
            Forecasting results can help the company plan machine operation, reduce unnecessary
            grid dependency and improve energy reliability. By knowing the expected Solar PV output,
            the company can make better decisions for smart energy management.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.subheader("Predicted Solar PV Output Trend")

        fig3, ax3 = plt.subplots(figsize=(12, 6))

        ax3.plot(
            df["Month"],
            df["Prediction_2026"],
            marker="o",
            linewidth=2,
            label="Prediction 2026"
        )

        ax3.axhline(
            y=average_prediction,
            linestyle="--",
            label="Average Prediction"
        )

        ax3.set_title("Predicted Solar PV Output Trend for 2026")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Predicted Solar PV Output")
        ax3.legend()
        ax3.grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig3)

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

        st.subheader("Project Overview")
        st.markdown(
            """
            <p style="color:#0f172a; font-size:16px; line-height:1.7;">
            This project focuses on forecasting Solar PV output using a deep learning approach.
            Solar PV energy is one of the renewable energy sources that can help reduce dependency
            on grid electricity, especially for industrial manufacturing companies that use a high
            amount of electrical energy in their daily operations.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="color:#0f172a; font-size:16px; line-height:1.7;">
            In this project, historical Solar PV data and environmental data are used to identify
            the pattern of solar energy generation. The forecasting model is developed using
            Long Short-Term Memory (LSTM), which is suitable for time-series forecasting.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <p style="color:#0f172a; font-size:16px; line-height:1.7;">
            The forecasting result is displayed through this Graphical User Interface (GUI).
            The GUI allows users to view the comparison between actual Solar PV output for 2025
            and predicted Solar PV output for 2026. It also displays performance metrics such as
            R² Score, MAE and RMSE to evaluate the model performance.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Problem Background")
        st.markdown(
            """
            <p style="color:#0f172a; font-size:16px; line-height:1.7;">
            Industrial manufacturing companies usually require a large amount of electricity to
            operate machines, equipment and production systems. Without accurate forecasting,
            it is difficult to estimate future solar energy generation. This may lead to inefficient
            energy planning and higher dependency on grid electricity.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Project Objectives")
        st.markdown(
            """
            <ol style="color:#0f172a; font-size:16px; line-height:1.7;">
                <li>To analyze environmental and Solar PV data to identify the pattern of solar energy generation.</li>
                <li>To develop a deep learning model using LSTM for forecasting Solar PV output.</li>
                <li>To apply the forecasting results in a simulated smart energy management system for better energy planning.</li>
            </ol>
            """,
            unsafe_allow_html=True
        )

        st.subheader("GUI Functions")
        st.markdown(
            """
            <ol style="color:#0f172a; font-size:16px; line-height:1.7;">
                <li>User sign in using name, email and position.</li>
                <li>Display Solar PV forecasting dataset.</li>
                <li>Display Actual 2025 vs Prediction 2026 graph.</li>
                <li>Display monthly comparison bar chart.</li>
                <li>Display R² Score, MAE and RMSE.</li>
                <li>Provide smart energy management insights.</li>
                <li>Save user login records automatically.</li>
                <li>Allow users to download forecasting result and login records.</li>
            </ol>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Smart Energy Management Application")
        st.markdown(
            """
            <p style="color:#0f172a; font-size:16px; line-height:1.7;">
            The forecasting result can help the industrial manufacturing company plan energy usage
            more effectively. For example, high-energy operations can be scheduled during periods
            with higher predicted Solar PV output. This can help optimize solar energy utilization,
            reduce unnecessary grid dependency and improve energy reliability.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.subheader("System Development Tool")
        st.markdown(
            """
            <p style="color:#0f172a; font-size:16px; line-height:1.7;">
            This GUI is developed using Python and Streamlit. Streamlit is used to create a
            web-based graphical user interface that can be accessed through a browser. The system
            is deployed using Streamlit Cloud so users can access the forecasting dashboard through
            an online link.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.subheader("System Flow")
        st.info(
            "User Sign In → Dashboard Overview → Forecasting Result → Performance Metrics → Energy Management Insight → Report Download"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    © 2026 Solar PV Output Forecasting System | Developed by Siti Irdina Umairah | Universiti Tun Hussein Onn Malaysia
</div>
""", unsafe_allow_html=True)
