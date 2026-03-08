import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from components import (
    brand_logo, api_status_badge, stat_card,
    prediction_result_card, page_header,
    churn_badge, empty_state
)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000") # Change to hosted URL after Docker

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────
# LOAD CSS
# ─────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ─────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────
def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200
    except:
        return False


def predict_single(payload: dict) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/churn/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def map_inputs(raw: dict) -> dict:
    internet = raw["internet_service"]
    contract = raw["contract_type"]
    payment = raw["payment_method"]

    return {
        "gender":                               int(raw["gender"] == "Male"),
        "SeniorCitizen":                        int(raw["senior_citizen"]),
        "Partner":                              int(raw["partner"]),
        "Dependents":                           int(raw["dependents"]),
        "tenure":                               raw["tenure"],
        "MultipleLines":                        int(raw["multiple_lines"]),
        "OnlineSecurity":                       int(raw["online_security"]),
        "OnlineBackup":                         int(raw["online_backup"]),
        "DeviceProtection":                     int(raw["device_protection"]),
        "TechSupport":                          int(raw["tech_support"]),
        "StreamingTV":                          int(raw["streaming_tv"]),
        "StreamingMovies":                      int(raw["streaming_movies"]),
        "PaperlessBilling":                     int(raw["paperless_billing"]),
        "MonthlyCharges":                       raw["monthly_charges"],
        "InternetService_Fiber_optic":          int(internet == "Fiber Optic"),
        "InternetService_No":                   int(internet == "No Internet"),
        "PaymentMethod_Bank_transfer_automatic": int(payment == "Bank Transfer"),
        "PaymentMethod_Credit_card_automatic":  int(payment == "Credit Card"),
        "PaymentMethod_Electronic_check":       int(payment == "Electronic Check"),
        "PaymentMethod_Mailed_check":           int(payment == "Mailed Check"),
        "Contract_One_year":                    int(contract == "One Year"),
        "Contract_Two_year":                    int(contract == "Two Year"),
    }


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown(brand_logo(), unsafe_allow_html=True)

    api_online = check_api_health()
    st.markdown(api_status_badge(api_online), unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📈 Predict", "📊 Dashboard", "📁 Batch Predict"],
        label_visibility="collapsed"
    )


# ─────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────
if page == "📈 Predict":

    st.markdown(page_header(
        "Customer Churn Predictor",
        "Enter customer details to predict churn probability"
    ), unsafe_allow_html=True)

    col_form, col_result = st.columns([2, 1], gap="large")

    with col_form:

        # Row 1 — Tenure + Monthly Charges
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (Months)", 0, 72, 24)
        with c2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0,
                max_value=200.0, value=65.0, step=0.5
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2 — Demographics
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            senior_citizen = st.checkbox("Senior Citizen")
        with c3:
            partner = st.checkbox("Partner")

        # Row 3 — Service toggles
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dependents = st.checkbox("Dependents")
        with c2:
            online_security = st.checkbox("Online Security")
        with c3:
            tech_support = st.checkbox("Tech Support")
        with c4:
            paperless_billing = st.checkbox("Paperless Billing", value=True)

        # Row 4 — More services
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            multiple_lines = st.checkbox("Multiple Lines")
        with c2:
            online_backup = st.checkbox("Online Backup")
        with c3:
            device_protection = st.checkbox("Device Protection")
        with c4:
            streaming_tv = st.checkbox("Streaming TV")

        streaming_movies = st.checkbox("Streaming Movies")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 5 — Dropdowns
        c1, c2, c3 = st.columns(3)
        with c1:
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber Optic", "No Internet"]
            )
        with c2:
            contract_type = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One Year", "Two Year"]
            )
        with c3:
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic Check", "Mailed Check",
                 "Bank Transfer", "Credit Card"]
            )

        st.markdown("<br>", unsafe_allow_html=True)

        predict_clicked = st.button("🔍 Predict Churn", use_container_width=True)

    # Result column
    with col_result:
        st.markdown("<br><br>", unsafe_allow_html=True)

        if predict_clicked:
            if not api_online:
                st.error("API is offline. Start FastAPI first.")
            else:
                raw = {
                    "gender": gender,
                    "senior_citizen": senior_citizen,
                    "partner": partner,
                    "dependents": dependents,
                    "tenure": tenure,
                    "monthly_charges": monthly_charges,
                    "multiple_lines": multiple_lines,
                    "online_security": online_security,
                    "online_backup": online_backup,
                    "device_protection": device_protection,
                    "tech_support": tech_support,
                    "streaming_tv": streaming_tv,
                    "streaming_movies": streaming_movies,
                    "paperless_billing": paperless_billing,
                    "internet_service": internet_service,
                    "contract_type": contract_type,
                    "payment_method": payment_method,
                }

                with st.spinner("Predicting..."):
                    result = predict_single(map_inputs(raw))

                if result:
                    st.markdown(
                        prediction_result_card(
                            result["will_churn"],
                            result["churn_probability"],
                            result["risk_level"]
                        ),
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                empty_state("Fill the form and click<br>Predict Churn", "🎯"),
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────
elif page == "📊 Dashboard":

    st.markdown(page_header(
        "Model Dashboard",
        "RandomForest model performance and feature insights"
    ), unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(stat_card("Model", "Random Forest", "🌲"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card("Features Used", "22", "📋"), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card("Status", "Live" if api_online else "Offline", "📡"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns([3, 2], gap="large")

    with chart_col1:
        # Feature Importance chart
        features = pd.DataFrame({
            "Feature": [
                "Tenure", "MonthlyCharges", "OnlineSecurity",
                "Contract_Two year", "InternetService_Fiber optic",
                "PaymentMethod_Electronic check", "Contract_One year",
                "TechSupport", "PaperlessBilling", "InternetService_No"
            ],
            "Importance (%)": [
                21.7, 12.8, 11.0, 9.7, 6.8,
                5.3, 4.5, 2.9, 2.5, 2.3
            ]
        }).sort_values("Importance (%)")

        fig = px.bar(
            features,
            x="Importance (%)",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color="Importance (%)",
            color_continuous_scale=[[0, "#1A3A2A"], [1, "#00FF94"]],
        )

        fig.update_layout(
            plot_bgcolor="#0D0D1A",
            paper_bgcolor="#0D0D1A",
            font=dict(color="#E8E8F0", family="IBM Plex Sans"),
            title=dict(font=dict(color="#FFFFFF", size=16)),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(
                gridcolor="#1A1A2E",
                showgrid=True,
                zeroline=False,
                tickfont=dict(color="#666680", size=11),
            ),
            yaxis=dict(
                gridcolor="#1A1A2E",
                showgrid=False,
                tickfont=dict(color="#AAAACC", size=12),
            ),
            height=380
        )

        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width='stretch')

    with chart_col2:
        # Churn Distribution donut
        fig2 = go.Figure(data=[go.Pie(
            labels=["No Churn", "Churn"],
            values=[73, 27],
            hole=0.65,
            marker=dict(
                colors=["#00FF94", "#FF4444"],
                line=dict(color="#0A0A0F", width=3)
            ),
            textfont=dict(color="#E8E8F0", size=13),
            hovertemplate="%{label}: %{value}%<extra></extra>"
        )])

        fig2.add_annotation(
            text="<b>Dataset</b><br>Distribution",
            x=0.5, y=0.5,
            font=dict(color="#AAAACC", size=12, family="IBM Plex Sans"),
            showarrow=False
        )

        fig2.update_layout(
            title=dict(
                text="Churn Distribution",
                font=dict(color="#FFFFFF", size=16)
            ),
            plot_bgcolor="#0D0D1A",
            paper_bgcolor="#0D0D1A",
            font=dict(color="#E8E8F0", family="IBM Plex Sans"),
            legend=dict(
            font=dict(color="#AAAACC"),
            bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=380
        )

        st.plotly_chart(fig2, width='stretch')

    # Model metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Model Metrics")

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("F1 Score", "0.62", "Cross-validated"),
        ("Precision", "0.64", "Churn class"),
        ("Recall", "0.61", "Churn class"),
        ("Accuracy", "0.79", "Overall"),
    ]

    for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.metric(label=label, value=value, delta=sub)


# ─────────────────────────────────────────
# PAGE: BATCH PREDICT
# ─────────────────────────────────────────
elif page == "📁 Batch Predict":

    st.markdown(page_header(
        "Batch Prediction",
        "Upload a CSV file to predict churn for multiple customers"
    ), unsafe_allow_html=True)

    # Expected columns info
    with st.expander("📋 Expected CSV columns"):
        st.code("""gender, SeniorCitizen, Partner, Dependents, tenure,
MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, PaperlessBilling,
MonthlyCharges, InternetService_Fiber optic, InternetService_No,
PaymentMethod_Bank transfer (automatic), PaymentMethod_Credit card (automatic),
PaymentMethod_Electronic check, PaymentMethod_Mailed check,
Contract_One year, Contract_Two year""", language="text")
        st.caption("All binary columns should be 0 or 1. MonthlyCharges is float.")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV with customer data to predict churn"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown(f"**{len(df)} customers loaded**")
        st.dataframe(
            df.head(5),
            use_container_width=True,
            hide_index=True
        )

        run_clicked = st.button("⚡ Run Predictions", use_container_width=True)

        if run_clicked:
            if not api_online:
                st.error("API is offline. Start FastAPI first.")
            else:
                results = []
                progress = st.progress(0, text="Running predictions...")

                for i, row in df.iterrows():
                    result = predict_single(row.to_dict())
                    if result:
                        results.append({
                            "will_churn": result["will_churn"],
                            "churn_probability": result["churn_probability"],
                            "risk_level": result["risk_level"]
                        })
                    else:
                        results.append({
                            "will_churn": None,
                            "churn_probability": None,
                            "risk_level": "error"
                        })
                    progress.progress(
                        (i + 1) / len(df),
                        text=f"Predicting customer {i+1}/{len(df)}..."
                    )

                progress.empty()

                # Merge results with original data
                result_df = df.copy()
                result_df["Will Churn"] = [r["will_churn"] for r in results]
                result_df["Churn Probability"] = [r["churn_probability"] for r in results]
                result_df["Risk Level"] = [r["risk_level"] for r in results]

                # Summary stats
                total = len(result_df)
                churned = result_df["Will Churn"].sum()
                high_risk = (result_df["Risk Level"] == "high").sum()

                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(stat_card("Total Customers", str(total), "👥"), unsafe_allow_html=True)
                with s2:
                    st.markdown(stat_card("Predicted Churn", str(churned), "⚠️"), unsafe_allow_html=True)
                with s3:
                    st.markdown(stat_card("High Risk", str(high_risk), "🔴"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Results")

                st.dataframe(df, width='stretch', hide_index=True)

                # Download
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
    else:
        st.markdown(
            empty_state("Upload a CSV file to get started", "📁"),
            unsafe_allow_html=True
        )
