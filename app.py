import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# PAGE Configure
st.set_page_config(
    page_title="Patient Health Record Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = os.path.join(
    "data",
    "patients_data_with_alerts(2).xlsx"
)

MODEL_PATH = os.path.join(
    "model",
    "random_forest_model.pkl"
)

FEATURE_PATH = os.path.join(
    "model",
    "model_features.pkl"
)

# CUSTOM CSS

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    color: #8b949e;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 10px;
    margin-bottom: 15px;
}

.metric-card {
    padding: 10px;
}

div[data-testid="stMetric"] {
    border-radius: 12px;
    padding: 15px;
    border: 1px solid rgba(128,128,128,0.20);
}

.alert-box {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    color: #888;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# LOAD DATA

@st.cache_data
def load_data():

    data = pd.read_excel(DATA_PATH)

    # Remove completely empty rows
    data = data.dropna(how="all")

    return data


# LOAD MODEL ONLY WHEN REQUIRED

@st.cache_resource
def load_ml_model():

    model = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURE_PATH):
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

if not os.path.exists(DATA_PATH):

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

st.markdown(
    '<div class="main-title">🏥 Patient Health Record Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Data Science Based Healthcare Analytics & Monitoring System'
    '</div>',
    unsafe_allow_html=True
)


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

filtered_df = df.copy()

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

    if selected_diseases:

        filtered_df = filtered_df[
            filtered_df[DISEASE_COL]
            .astype(str)
            .isin(selected_diseases)
        ]


# DASHBOARD OVERVIEW

if page == "📊 Dashboard Overview":

    st.markdown(
        '<div class="section-title">📊 Healthcare Overview</div>',
        unsafe_allow_html=True
    )

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
                use_container_width=True
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
                use_container_width=True
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
            use_container_width=True,
            hide_index=True
        )


# PATIENT EXPLORER

elif page == "🔎 Patient Explorer":

    st.markdown(
        '<div class="section-title">🔎 Patient Explorer</div>',
        unsafe_allow_html=True
    )

    if PATIENT_COL is None:

        st.warning(
            "Patient ID column was not detected."
        )

    else:

        patient_ids = (
            df[PATIENT_COL]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_patient = st.selectbox(
            "Select Patient",
            sorted(patient_ids)
        )

        patient = df[
            df[PATIENT_COL].astype(str)
            == selected_patient
        ]

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
                use_container_width=True
            )


# VITAL ANALYTICS

elif page == "❤️ Vital Analytics":

    st.markdown(
        '<div class="section-title">❤️ Vital Sign Analytics</div>',
        unsafe_allow_html=True
    )

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
                use_container_width=True
            )

        with col2:

            fig = px.box(
                filtered_df,
                y=selected_vital,
                title=f"{selected_vital} Box Plot"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("📈 Statistics")

        st.dataframe(
            filtered_df[selected_vital]
            .describe()
            .to_frame("Value"),
            use_container_width=True
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
            use_container_width=True
        )


# DISEASE ANALYTICS

elif page == "🩺 Disease Analytics":

    st.markdown(
        '<div class="section-title">🩺 Disease Analytics</div>',
        unsafe_allow_html=True
    )

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
                use_container_width=True
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
                use_container_width=True
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
            use_container_width=True
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
                use_container_width=True,
                hide_index=True
            )

# HEALTH ALERTS

elif page == "⚠️ Health Alerts":

    st.markdown(
        '<div class="section-title">⚠️ Health Alerts & Monitoring</div>',
        unsafe_allow_html=True
    )

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
                use_container_width=True
            )

        with col2:

            st.dataframe(
                fall_counts,
                use_container_width=True,
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
            use_container_width=True
        )

        st.dataframe(
            alert_counts,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No alert columns were detected."
        )

# MACHINE LEARNING PREDICTION

elif page == "🤖 ML Prediction":

    st.markdown(
        '<div class="section-title">🤖 Machine Learning Prediction</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The prediction is generated by the trained Random Forest "
        "model. It is intended for academic/data-analysis purposes "
        "and is not a medical diagnosis."
    )

    if not os.path.exists(MODEL_PATH):

        st.warning(
            "Random Forest model is not available yet."
        )

        st.write(
            "Place random_forest_model.pkl inside the model folder."
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
            use_container_width=True
        )

        if predict:

            with st.spinner(
                "Loading model and generating prediction..."
            ):

                try:

                    model, features = load_ml_model()

                    patient_data = pd.DataFrame(
                        {
                        "Heart Rate (bpm)": [heart_rate],
                        "SpO2 Level (%)": [spo2],
                        "Systolic Blood Pressure (mmHg)": [systolic_bp],
                        "Diastolic Blood Pressure (mmHg)": [diastolic_bp],
                        "Body Temperature (°C)": [temperature]
                        }
                                                )

                    # Ensure correct feature order
                    patient_data = patient_data[features]

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
                        use_container_width=True
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
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
                        use_container_width=True
                    )

                except Exception as error:

                    st.error(
                        "Prediction could not be generated."
                    )

                    st.code(
                        str(error)
                    )


# DATASET EXPLORER

elif page == "📋 Dataset Explorer":

    st.markdown(
        '<div class="section-title">📋 Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore the underlying patient healthcare dataset."
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

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

    st.markdown("---")

    st.subheader("Patient Records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
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
        use_container_width=True,
        hide_index=True
    )


# FOOTER

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    Patient Healthcare Intelligence<br>
    <small>
    <i>Turning Patient Data into Actionable Insights</i>
    </small><br>
    <strong>Designed & Developed by Aman Chaurasiya</strong> • Data Science & Machine Learning • Data Analytics<br>
    <small>© 2026 Aman Chaurasiya. All Rights Reserved. Unauthorized reproduction or distribution is prohibited.</small><br>
    </div>
    """,
    unsafe_allow_html=True
)