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
<style>
.stApp {
    background-color: #f8fafc;
}

.main-title {
    background: linear-gradient(90deg, #0f172a, #ca8a04);
    padding: 36px;
    border-radius: 22px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.20);
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
}

.prepared {
    font-size: 20px;
    color: #ffffff;
    font-weight: 500;
}

.section-card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.info-box {
    background-color: #ecfeff;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #0891b2;
    color: #164e63;
    font-size: 16px;
}

.warning-box {
    background-color: #fffbeb;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #ca8a04;
    color: #713f12;
    font-size: 16px;
}

.success-box {
    background-color: #ecfdf5;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #059669;
    color: #064e3b;
    font-size: 16px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
}

.login-title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}

.login-subtitle {
    text-align: center;
    font-size: 16px;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER WITH CENTER LOGO
# ============================================================

logo_file = "uthm_logo.jpg.new"

st.markdown("""
<div style="
    text-align: center;
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
">
""", unsafe_allow_html=True)

if os.path.isfile(logo_file):
    st.image(logo_file, width=380)
else:
    st.error("Logo file not found. Please check the logo filename in GitHub.")

st.markdown("</div>", unsafe_allow_html=True)

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

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.markdown('<p class="login-title">User Sign In</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="login-subtitle">Please enter your details before accessing the Solar PV forecasting dashboard.</p>',
        unsafe_allow_html=True
    )

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

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        This GUI system allows users to view the Solar PV forecasting results,
        performance metrics and energy management insights through an interactive dashboard.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN GUI DASHBOARD
# ============================================================

else:

    df = load_forecast_data()

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
            This dashboard shows the monthly average Solar PV output based on actual data for 2025
            and predicted data for 2026. The result can help users understand the solar generation
            pattern and support smart energy management decisions.
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # FORECASTING RESULT PAGE
    # ========================================================

    elif menu == "Forecasting Result":

        st.header("Forecasting Result")

        st.markdown("""
        <div class="info-box">
            This section displays the comparison between Actual 2025 and Prediction 2026 Solar PV output.
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
            activities when solar generation is expected to be higher.
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown(f"""
        <div class="warning-box">
            <b>2. Low Solar Generation Period</b><br>
            The lowest predicted Solar PV output occurs in <b>{lowest_month}</b>.
            During this period, the company may need to manage energy usage carefully
            and depend more on grid electricity or backup energy sources.
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

        st.subheader("Monthly Forecast Trend")

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
        st.success("Deep Learning-Based Forecasting of Solar PV Output for Indusrial Manufacturing Company")

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
        4. Display monthly comparison bar chart.  
        5. Calculate and display R² Score, MAE and RMSE.  
        6. Provide smart energy management insights.  
        7. Save user login records automatically.  
        8. Allow user to download forecasting result and login records.
        """)

        st.subheader("System Development Tool")
        st.write("""
        This GUI is developed using Python and Streamlit. Streamlit is used to create
        a web-based graphical user interface that can be accessed through a browser.
        """)

        st.subheader("System Flow")
        st.info("""
        User Sign In → Dashboard Overview → Forecasting Result → Performance Metrics
        → Energy Management Insight → Report Download
        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    © 2026 Solar PV Output Forecasting System | Developed by Siti Irdina Umairah | Universiti Tun Hussein Onn Malaysia
</div>
""", unsafe_allow_html=True)
