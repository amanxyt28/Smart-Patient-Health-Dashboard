import logging
import os
import tempfile
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from health_utils import (
    build_prediction_frame,
    download_verified_artifact,
    filter_patient_records,
    paginate_frame,
    sha256_file,
)

# PAGE CONFIG
st.set_page_config(
    page_title="Patient Health Record Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "patients_data_with_alerts(2).xlsx"
MODEL_PATH = BASE_DIR / "model" / "random_forest_model.pkl"
FEATURE_PATH = BASE_DIR / "model" / "model_features.pkl"
LOGGER = logging.getLogger(__name__)


def get_deployment_setting(name):
    try:
        return st.secrets.get(name) or os.getenv(name)
    except FileNotFoundError:
        return os.getenv(name)


def get_model_path():
    if MODEL_PATH.is_file():
        return MODEL_PATH

    model_url = get_deployment_setting("MODEL_URL")
    model_sha256 = get_deployment_setting("MODEL_SHA256")
    if not model_url or not model_sha256:
        raise FileNotFoundError(
            "Set MODEL_URL and MODEL_SHA256 in Streamlit secrets to enable predictions."
        )

    cached_path = (
        Path(tempfile.gettempdir())
        / "patient-health-dashboard"
        / "random_forest_model.pkl"
    )
    if not cached_path.is_file() or sha256_file(cached_path) != model_sha256.lower():
        return download_verified_artifact(
            model_url,
            model_sha256,
            cached_path,
        )
    return cached_path

# LOAD DATA

@st.cache_data(max_entries=2)
def load_data():

    data = pd.read_excel(DATA_PATH)

    # Remove completely empty rows
    data = data.dropna(how="all")

    return data


# LOAD MODEL ONLY WHEN REQUIRED

@st.cache_resource(max_entries=1)
def load_ml_model():

    model = joblib.load(get_model_path())

    if FEATURE_PATH.exists():
        features = joblib.load(FEATURE_PATH)
    else:
        features = [
            "Heart Rate (bpm)",
            "SpO2 Level (%)",
            "Systolic Blood Pressure (mmHg)",
            "Diastolic Blood Pressure (mmHg)",
            "Body Temperature (°C)"
        ]

    return model, features


# FILE CHECK

if not DATA_PATH.exists():

    st.error(
        f"""
        Dataset not found.

        Expected location:

        {DATA_PATH}
        """
    )

    st.stop()


# LOAD DATASET

df = load_data()


# COLUMN DETECTION

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


PATIENT_COL = find_column([
    "Patient Number",
    "Patient ID",
    "Patient_ID",
    "PatientID",
    "ID"
])

HEART_COL = find_column([
    "Heart Rate (bpm)",
    "Heart Rate",
    "Heart_Rate",
    "Heart rate"
])

SPO2_COL = find_column([
    "SpO2 Level (%)",
    "SpO2",
    "SpO2 Level",
    "SPO2"
])

SYS_BP_COL = find_column([
    "Systolic Blood Pressure (mmHg)",
    "Systolic Blood Pressure",
    "Systolic BP",
    "Systolic_BP"
])

DIA_BP_COL = find_column([
    "Diastolic Blood Pressure (mmHg)",
    "Diastolic Blood Pressure",
    "Diastolic BP",
    "Diastolic_BP"
])

TEMP_COL = find_column([
    "Body Temperature (°C)",
    "Body Temperature",
    "Temperature",
    "Body_Temperature"
])

ACCURACY_COL = find_column([
    "Data Accuracy (%)",
    "Data Accuracy",
    "Data_Accuracy"
])

DISEASE_COL = find_column([
    "Predicted Disease",
    "Disease",
    "Diagnosis",
    "Predicted_Disease"
])

FALL_COL = find_column([
    "Fall Detection",
    "Fall_Detection",
    "Fall detection"
])


# HEADER

st.title("Patient Health Record Dashboard")
st.caption("Data science based healthcare analytics and monitoring system")


# SIDEBAR

st.sidebar.title("🏥 Patient Health")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "📊 Dashboard Overview",
        "🔎 Patient Explorer",
        "❤️ Vital Analytics",
        "🩺 Disease Analytics",
        "⚠️ Health Alerts",
        "🤖 ML Prediction",
        "📋 Dataset Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Records loaded: {len(df):,}"
)

st.sidebar.caption(
    f"Columns: {len(df.columns)}"
)


# FILTER FUNCTION

filtered_df = df

if DISEASE_COL is not None:

    diseases = sorted(
        df[DISEASE_COL]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_diseases = st.sidebar.multiselect(
        "Disease Filter",
        diseases,
        default=diseases
    )

    if selected_diseases and len(selected_diseases) < len(diseases):

        filtered_df = filtered_df.loc[
            filtered_df[DISEASE_COL]
            .astype(str)
            .isin(selected_diseases)
        ]


# DASHBOARD OVERVIEW

if page == "📊 Dashboard Overview":

    st.header("Healthcare overview")

    st.write(
        "Interactive summary of the available patient health records."
    )
    # KPI CARDS

    total_patients = len(filtered_df)

    disease_categories = (
        filtered_df[DISEASE_COL].nunique()
        if DISEASE_COL else 0
    )

    avg_hr = (
        filtered_df[HEART_COL].mean()
        if HEART_COL else 0
    )

    avg_spo2 = (
        filtered_df[SPO2_COL].mean()
        if SPO2_COL else 0
    )

    avg_temp = (
        filtered_df[TEMP_COL].mean()
        if TEMP_COL else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "👥 Patients",
        f"{total_patients:,}"
    )

    c2.metric(
        "🩺 Disease Categories",
        disease_categories
    )

    c3.metric(
        "❤️ Avg Heart Rate",
        f"{avg_hr:.1f} bpm"
    )

    c4.metric(
        "🫁 Avg SpO₂",
        f"{avg_spo2:.1f}%"
    )

    c5.metric(
        "🌡️ Avg Temperature",
        f"{avg_temp:.1f} °C"
    )

    st.markdown("---")

    # DISEASE DISTRIBUTION

    if DISEASE_COL:

        col1, col2 = st.columns(2)

        disease_counts = (
            filtered_df[DISEASE_COL]
            .value_counts()
            .reset_index()
        )

        disease_counts.columns = [
            "Disease",
            "Patients"
        ]

        with col1:

            fig = px.bar(
                disease_counts,
                x="Disease",
                y="Patients",
                title="Disease Distribution",
                text="Patients"
            )

            fig.update_layout(
                xaxis_title="Disease",
                yaxis_title="Number of Patients"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.pie(
                disease_counts,
                names="Disease",
                values="Patients",
                title="Disease Proportion",
                hole=0.4
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

    # VITAL SUMMARY

    st.subheader("📋 Vital Statistics")

    vital_data = {}

    if HEART_COL:
        vital_data["Heart Rate"] = filtered_df[HEART_COL]

    if SPO2_COL:
        vital_data["SpO₂"] = filtered_df[SPO2_COL]

    if SYS_BP_COL:
        vital_data["Systolic BP"] = filtered_df[SYS_BP_COL]

    if DIA_BP_COL:
        vital_data["Diastolic BP"] = filtered_df[DIA_BP_COL]

    if TEMP_COL:
        vital_data["Temperature"] = filtered_df[TEMP_COL]

    if vital_data:

        stats = pd.DataFrame({
            "Parameter": list(vital_data.keys()),
            "Average": [
                values.mean()
                for values in vital_data.values()
            ],
            "Minimum": [
                values.min()
                for values in vital_data.values()
            ],
            "Maximum": [
                values.max()
                for values in vital_data.values()
            ],
            "Std Dev": [
                values.std()
                for values in vital_data.values()
            ]
        })

        st.dataframe(
            stats,
            width="stretch",
            hide_index=True
        )


# PATIENT EXPLORER

elif page == "🔎 Patient Explorer":

    st.header("Patient explorer")

    if PATIENT_COL is None:

        st.warning(
            "Patient ID column was not detected."
        )

    else:

        with st.form("patient_search"):
            selected_patient = st.text_input(
                "Patient number",
                placeholder="Enter a patient number",
            )
            patient_submitted = st.form_submit_button("Find patient")

        patient = df.iloc[0:0]
        if patient_submitted:
            if selected_patient.strip():
                patient = filter_patient_records(
                    df,
                    PATIENT_COL,
                    selected_patient,
                )
                if patient.empty:
                    st.warning("No record was found for that patient number.")
            else:
                st.warning("Enter a patient number to search.")

        if not patient.empty:

            record = patient.iloc[0]

            st.success(
                f"Patient {selected_patient} selected"
            )

            # Vital cards

            c1, c2, c3, c4, c5 = st.columns(5)

            if HEART_COL:
                c1.metric(
                    "❤️ Heart Rate",
                    f"{record[HEART_COL]} bpm"
                )

            if SPO2_COL:
                c2.metric(
                    "🫁 SpO₂",
                    f"{record[SPO2_COL]}%"
                )

            if SYS_BP_COL:
                c3.metric(
                    "🩸 Systolic BP",
                    f"{record[SYS_BP_COL]} mmHg"
                )

            if DIA_BP_COL:
                c4.metric(
                    "🩸 Diastolic BP",
                    f"{record[DIA_BP_COL]} mmHg"
                )

            if TEMP_COL:
                c5.metric(
                    "🌡️ Temperature",
                    f"{record[TEMP_COL]} °C"
                )

            st.markdown("---")

            # Disease

            if DISEASE_COL:

                st.subheader("🩺 Patient Disease Category")

                st.info(
                    str(record[DISEASE_COL])
                )

            # Full record

            st.subheader("📋 Complete Patient Record")

            st.dataframe(
                patient.T,
                width="stretch"
            )


# VITAL ANALYTICS

elif page == "❤️ Vital Analytics":

    st.header("Vital sign analytics")

    vital_options = []

    if HEART_COL:
        vital_options.append(HEART_COL)

    if SPO2_COL:
        vital_options.append(SPO2_COL)

    if SYS_BP_COL:
        vital_options.append(SYS_BP_COL)

    if DIA_BP_COL:
        vital_options.append(DIA_BP_COL)

    if TEMP_COL:
        vital_options.append(TEMP_COL)

    if not vital_options:

        st.error("No vital columns detected.")

    else:

        selected_vital = st.selectbox(
            "Select Vital Parameter",
            vital_options
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                filtered_df,
                x=selected_vital,
                nbins=35,
                title=f"{selected_vital} Distribution"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.box(
                filtered_df,
                y=selected_vital,
                title=f"{selected_vital} Box Plot"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        st.subheader("📈 Statistics")

        st.dataframe(
            filtered_df[selected_vital]
            .describe()
            .to_frame("Value"),
            width="stretch"
        )

    # CORRELATION

    st.markdown("---")

    st.subheader("🔗 Vital Parameter Correlation")

    numeric_vitals = []

    for column in [
        HEART_COL,
        SPO2_COL,
        SYS_BP_COL,
        DIA_BP_COL,
        TEMP_COL
    ]:

        if column:
            numeric_vitals.append(column)

    if len(numeric_vitals) >= 2:

        correlation = filtered_df[
            numeric_vitals
        ].corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# DISEASE ANALYTICS

elif page == "🩺 Disease Analytics":

    st.header("Disease analytics")

    if DISEASE_COL is None:

        st.error(
            "Disease column was not detected."
        )

    else:

        disease_counts = (
            filtered_df[DISEASE_COL]
            .value_counts()
            .reset_index()
        )

        disease_counts.columns = [
            "Disease",
            "Patients"
        ]

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                disease_counts,
                x="Disease",
                y="Patients",
                color="Disease",
                title="Patients by Disease"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.pie(
                disease_counts,
                names="Disease",
                values="Patients",
                title="Disease Distribution",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        # VITALS BY DISEASE

        st.subheader(
            "📊 Vital Parameters by Disease"
        )

        selected_vital = st.selectbox(
            "Select Parameter",
            [
                col for col in [
                    HEART_COL,
                    SPO2_COL,
                    SYS_BP_COL,
                    DIA_BP_COL,
                    TEMP_COL
                ]
                if col
            ]
        )

        fig = px.box(
            filtered_df,
            x=DISEASE_COL,
            y=selected_vital,
            color=DISEASE_COL,
            title=f"{selected_vital} by Disease"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # AVERAGE TABLE

        group_columns = [
            col for col in [
                HEART_COL,
                SPO2_COL,
                SYS_BP_COL,
                DIA_BP_COL,
                TEMP_COL
            ]
            if col
        ]

        if group_columns:

            disease_summary = (
                filtered_df
                .groupby(DISEASE_COL)[group_columns]
                .mean()
                .round(2)
                .reset_index()
            )

            st.subheader(
                "Average Vital Parameters"
            )

            st.dataframe(
                disease_summary,
                width="stretch",
                hide_index=True
            )

# HEALTH ALERTS

elif page == "⚠️ Health Alerts":

    st.header("Health alerts and monitoring")

    st.info(
        "Alerts shown here are the alert classifications "
        "present in the dataset. They should not be interpreted "
        "as clinical diagnoses."
    )

    # FALL DETECTION

    if FALL_COL:

        st.subheader("🚨 Fall Detection")

        fall_counts = (
            filtered_df[FALL_COL]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        fall_counts.columns = [
            "Status",
            "Patients"
        ]

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                fall_counts,
                names="Status",
                values="Patients",
                title="Fall Detection Distribution"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            st.dataframe(
                fall_counts,
                width="stretch",
                hide_index=True
            )

    # ALERT COLUMNS

    alert_columns = [
        column
        for column in df.columns
        if "Alert" in column
        or "alert" in column
    ]

    if alert_columns:

        st.subheader("📢 Alert Summary")

        cols = st.columns(
            min(len(alert_columns), 4)
        )

        for index, column in enumerate(alert_columns):

            alert_count = (
                filtered_df[column]
                .astype(str)
                .str.lower()
                .str.contains(
                    "alert|warning|high|low|yes|true",
                    regex=True
                )
                .sum()
            )

            cols[index % len(cols)].metric(
                column,
                f"{alert_count:,}"
            )

        st.markdown("---")

        selected_alert = st.selectbox(
            "Select Alert Type",
            alert_columns
        )

        alert_counts = (
            filtered_df[selected_alert]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        alert_counts.columns = [
            "Status",
            "Patients"
        ]

        fig = px.bar(
            alert_counts,
            x="Status",
            y="Patients",
            color="Status",
            title=selected_alert
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.dataframe(
            alert_counts,
            width="stretch",
            hide_index=True
        )

    else:

        st.warning(
            "No alert columns were detected."
        )

# MACHINE LEARNING PREDICTION

elif page == "🤖 ML Prediction":

    st.header("Machine learning prediction")

    st.info(
        "The prediction is generated by the trained Random Forest "
        "model. It is intended for academic/data-analysis purposes "
        "and is not a medical diagnosis."
    )

    has_model_source = MODEL_PATH.is_file() or bool(
        get_deployment_setting("MODEL_URL")
        and get_deployment_setting("MODEL_SHA256")
    )

    if not has_model_source:

        st.warning(
            "Prediction model is not configured for this deployment."
        )

        st.write(
            "Add MODEL_URL and MODEL_SHA256 to Streamlit secrets, or place "
            "random_forest_model.pkl inside the model folder."
        )

    else:

        st.write(
            "Enter the five health parameters used during model training."
        )

        # INPUTS

        c1, c2 = st.columns(2)

        with c1:

            heart_rate = st.number_input(
                "❤️ Heart Rate (bpm)",
                min_value=20.0,
                max_value=250.0,
                value=80.0,
                step=1.0
            )

            spo2 = st.number_input(
                "🫁 SpO₂ Level (%)",
                min_value=50.0,
                max_value=100.0,
                value=97.0,
                step=0.5
            )

            systolic_bp = st.number_input(
                "🩸 Systolic BP (mmHg)",
                min_value=50.0,
                max_value=250.0,
                value=120.0,
                step=1.0
            )

        with c2:

            diastolic_bp = st.number_input(
                "🩸 Diastolic BP (mmHg)",
                min_value=30.0,
                max_value=180.0,
                value=80.0,
                step=1.0
            )

            temperature = st.number_input(
                "🌡️ Body Temperature (°C)",
                min_value=30.0,
                max_value=45.0,
                value=36.8,
                step=0.1
            )

        predict = st.button(
            "🔮 Generate Prediction",
            type="primary",
            width="stretch"
        )

        if predict:

            with st.spinner(
                "Loading model and generating prediction..."
            ):

                try:

                    model, features = load_ml_model()

                    input_values = {
                        "Heart Rate (bpm)": heart_rate,
                        "SpO2 Level (%)": spo2,
                        "Systolic Blood Pressure (mmHg)": systolic_bp,
                        "Diastolic Blood Pressure (mmHg)": diastolic_bp,
                        "Body Temperature (°C)": temperature,
                    }
                    patient_data = build_prediction_frame(input_values, features)

                    prediction = model.predict(
                        patient_data
                    )[0]

                    probabilities = model.predict_proba(
                        patient_data
                    )[0]

                    max_probability = probabilities.max()

                    st.markdown("---")

                    # RESULT

                    st.subheader(
                        "🎯 Prediction Result"
                    )

                    r1, r2 = st.columns(2)

                    r1.metric(
                        "Predicted Category",
                        str(prediction)
                    )

                    r2.metric(
                        "Highest Model Probability",
                        f"{max_probability * 100:.2f}%"
                    )
                    if max_probability < 0.50:

                        st.warning(
                            "The model shows relatively low confidence "
                            "for this input."
                        )

                    elif max_probability < 0.75:

                        st.info(
                            "The model shows moderate confidence "
                            "for this input."
                        )

                    else:

                        st.success(
                            "The model shows relatively high confidence "
                            "for this input."
                        )

                    # PROBABILITIES

                    probability_df = pd.DataFrame({
                        "Disease": model.classes_,
                        "Probability (%)": probabilities * 100
                    })

                    probability_df = (
                        probability_df
                        .sort_values(
                            "Probability (%)",
                            ascending=False
                        )
                    )

                    st.subheader(
                        "📊 Prediction Probabilities"
                    )

                    fig = px.bar(
                        probability_df,
                        x="Disease",
                        y="Probability (%)",
                        text="Probability (%)",
                        title="Random Forest Prediction Probabilities"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside"
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch"
                    )

                    st.dataframe(
                        probability_df,
                        width="stretch",
                        hide_index=True
                    )

                    # FEATURE IMPORTANCE

                    st.subheader(
                        "🔍 Model Feature Importance"
                    )

                    importance_df = pd.DataFrame({
                        "Feature": features,
                        "Importance": model.feature_importances_
                    }).sort_values(
                        "Importance",
                        ascending=True
                    )

                    fig = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Random Forest Feature Importance"
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch"
                    )

                except Exception:
                    LOGGER.exception("Prediction could not be generated")
                    st.error(
                        "Prediction could not be generated. Check that the model "
                        "and feature files are valid, then try again."
                    )


# DATASET EXPLORER

elif page == "📋 Dataset Explorer":

    st.header("Dataset explorer")

    st.write(
        "Explore the underlying patient healthcare dataset."
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Columns",
        len(filtered_df.columns)
    )

    c3.metric(
        "Duplicate Rows",
        int(filtered_df.duplicated().sum())
    )

    if ACCURACY_COL:
        c4.metric(
            "Mean Data Accuracy",
            f"{filtered_df[ACCURACY_COL].mean():.1f}%"
        )

    st.markdown("---")

    st.subheader("Patient Records")

    page_size = st.selectbox(
        "Rows per page",
        [50, 100, 250, 500],
        index=1,
    )
    page_count = max(1, (len(filtered_df) + page_size - 1) // page_size)
    page_number = st.number_input(
        "Page number",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
    )
    first_row = (page_number - 1) * page_size
    last_row = min(first_row + page_size, len(filtered_df))
    if len(filtered_df):
        st.caption(
            f"Showing rows {first_row + 1:,}-{last_row:,} "
            f"of {len(filtered_df):,}"
        )
    else:
        st.info("No patient records match the selected filters.")

    st.dataframe(
        paginate_frame(filtered_df, page_number, page_size),
        width="stretch",
        height=500
    )

    # MISSING VALUES
    
    st.subheader("🔎 Data Quality")

    missing = (
        filtered_df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values"
    ]

    missing["Missing %"] = (
        missing["Missing Values"]
        / len(filtered_df)
        * 100
    ).round(2)

    st.dataframe(
        missing,
        width="stretch",
        hide_index=True
    )


# FOOTER

st.markdown("---")

st.subheader("Patient healthcare intelligence")
st.caption("Turning patient data into actionable insights")
st.caption("Designed and developed by Aman Chaurasiya | Data science, machine learning, and analytics")