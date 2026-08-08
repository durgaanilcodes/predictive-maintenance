import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 650;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"

DATA_PATH = BASE_DIR.parent / "enhanced_dataset.csv"


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

model = None

if MODEL_PATH.exists():

    try:
        model = joblib.load(MODEL_PATH)

    except Exception as e:
        model = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔧 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Project Overview",
        "📊 Data Overview",
        "🤖 Model Performance",
        "📈 Threshold Analysis",
        "🔊 Noise Analysis",
        "📄 Final Report"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Contextual Predictive Maintenance

**Week 1**
Data Exploration

**Week 2**
Contextual Feature Engineering

**Week 3**
LightGBM Model Development

**Week 4**
Noise Validation & Threshold Analysis
""")

st.sidebar.markdown("---")

st.sidebar.success("👩‍💻 Team Lead: Durga Anil")

st.sidebar.markdown("""
### 👥 Team Members

- Durga Anil — Team Lead
- Binit Binu
- Krutika Thakur
- Priyadharshini
""")


# ============================================================
# VERIFIED PROJECT METRICS
# ============================================================

accuracy = 0.989000
precision = 0.767442
recall = 0.970588
f1_score = 0.857143


# ============================================================
# PAGE 1 — PROJECT OVERVIEW
# ============================================================

if page == "🏠 Project Overview":

    st.markdown(
        '<div class="main-title">🔧 Contextual Predictive Maintenance Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based Predictive Maintenance using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1_score * 100:.2f}%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.header("📌 Project Overview")

    st.write("""
    This project develops a machine learning-based predictive
    maintenance system designed to identify potential machine
    failures before breakdown occurs.

    The project focuses on contextual predictive maintenance by
    combining machine telemetry with contextual information and
    applying machine learning techniques for failure prediction.
    """)

    st.info("""
    🎯 **Main Objective**

    Build a robust predictive maintenance framework that can
    support proactive maintenance decisions and improve model
    reliability under imbalanced and noisy data conditions.
    """)

    # --------------------------------------------------------
    # PROJECT TECHNOLOGY
    # --------------------------------------------------------

    st.header("🧠 Technologies & Techniques")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### Machine Learning

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - LightGBM
        - SMOTE
        """)

    with col2:

        st.markdown("""
        ### Analysis

        - Exploratory Data Analysis
        - Feature Engineering
        - Contextual Data Fusion
        - 5-Fold Stratified Cross Validation
        - Threshold Tuning
        - Noise Sensitivity Analysis
        """)

    # --------------------------------------------------------
    # WEEKLY WORK
    # --------------------------------------------------------

    st.header("📅 Four-Week Project Journey")

    with st.expander("✅ Week 1 — Data Exploration"):

        st.write("""
        - Dataset ingestion
        - Data exploration
        - Signal analysis
        - Rolling statistics
        - Baseline feature creation
        """)

    with st.expander("✅ Week 2 — Contextual Feature Engineering"):

        st.write("""
        - Contextual data fusion
        - Feature engineering
        - Environmental/contextual feature analysis
        - Feature relationship study
        - Documentation
        """)

    with st.expander("✅ Week 3 — Model Development"):

        st.write("""
        - Imbalanced classification
        - SMOTE
        - LightGBM classifier
        - 5-Fold Stratified Cross Validation
        - Model evaluation
        """)

    with st.expander("✅ Week 4 — Noise & Threshold Analysis"):

        st.write("""
        - Precision-Recall analysis
        - Threshold tuning
        - Performance comparison
        - Noise sensitivity analysis
        - Final visualization and documentation
        """)

    st.markdown("---")

    st.success("🚀 Predictive Maintenance Project Dashboard")


# ============================================================
# PAGE 2 — DATA OVERVIEW
# ============================================================

elif page == "📊 Data Overview":

    st.title("📊 Dataset Overview")

    st.write("""
    The project uses the **AI4I 2020 Predictive Maintenance Dataset**
    as the foundation for predictive maintenance analysis.
    """)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if DATA_PATH.exists():

        try:

            df = pd.read_csv(DATA_PATH)

            st.success("✅ Enhanced dataset loaded successfully.")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Rows",
                f"{df.shape[0]:,}"
            )

            col2.metric(
                "Total Columns",
                df.shape[1]
            )

            col3.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

            col4.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

            st.markdown("---")

            st.subheader("📋 Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("📈 Numerical Statistics")

            st.dataframe(
                df.describe(),
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("🧾 Dataset Columns")

            columns_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values
            })

            st.dataframe(
                columns_df,
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("📊 Missing Values by Column")

            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isnull().sum().values
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        except Exception as e:

            st.error("Unable to read the dataset.")

            st.code(str(e))

    else:

        st.warning(
            "enhanced_dataset.csv was not found in the repository."
        )

        st.info("""
        Expected location:

        predictive-maintenance/
        ├── enhanced_dataset.csv
        │
        └── streamlit_app/
            └── app.py
        """)


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write("""
    The predictive maintenance model uses LightGBM for classification
    with SMOTE-based handling of class imbalance and stratified
    cross-validation.
    """)

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.subheader("🧠 Model Details")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        **Algorithm**

        LightGBM Classifier

        **Imbalance Handling**

        SMOTE
        """)

    with col2:

        st.markdown("""
        **Validation**

        5-Fold Stratified Cross Validation

        **Problem Type**

        Binary Classification
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1_score * 100:.2f}%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.subheader("📋 Classification Performance")

    performance_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1_score
        ]
    })

    performance_display = performance_df.copy()

    performance_display["Score"] = (
        performance_display["Score"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        performance_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # PERFORMANCE CHART
    # --------------------------------------------------------

    st.subheader("📈 Model Performance Comparison")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        performance_df["Metric"],
        performance_df["Score"]
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Score")

    ax.set_title(
        "LightGBM Model Performance"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------------

    st.subheader("🔍 Observations")

    st.write("""
    The evaluation results indicate strong predictive maintenance
    classification performance.

    The recall value is particularly high, indicating that the model
    identifies a large proportion of the actual failure cases.

    Precision and F1-score provide additional information about the
    trade-off between false alarms and missed failures.
    """)

    st.success(
        "The documented model evaluation demonstrates strong "
        "classification capability for predictive maintenance."
    )


# ============================================================
# PAGE 4 — THRESHOLD ANALYSIS
# ============================================================

elif page == "📈 Threshold Analysis":

    st.title("📈 Threshold Tuning Analysis")

    st.write("""
    Threshold tuning evaluates how Precision, Recall and F1 Score
    change as the classification probability threshold is adjusted.
    """)

    # --------------------------------------------------------
    # VERIFIED THRESHOLD RESULTS
    # --------------------------------------------------------

    threshold_df = pd.DataFrame({

        "Threshold": [
            0.3,
            0.4,
            0.5,
            0.6,
            0.7
        ],

        "Precision": [
            0.616822,
            0.717391,
            0.767442,
            0.804878,
            0.868421
        ],

        "Recall": [
            0.970588,
            0.970588,
            0.970588,
            0.970588,
            0.970588
        ],

        "F1 Score": [
            0.754286,
            0.825000,
            0.857143,
            0.880000,
            0.916667
        ]
    })

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("📋 Threshold Comparison")

    display_df = threshold_df.copy()

    for column in [
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader("📊 Threshold vs Performance")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="o",
        linewidth=2,
        label="Precision"
    )

    ax.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        marker="s",
        linewidth=2,
        label="Recall"
    )

    ax.plot(
        threshold_df["Threshold"],
        threshold_df["F1 Score"],
        marker="^",
        linewidth=2,
        label="F1 Score"
    )

    ax.set_xlabel("Classification Threshold")

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    ax.set_title(
        "Threshold Comparison"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # BEST F1
    # --------------------------------------------------------

    best_row = threshold_df.loc[
        threshold_df["F1 Score"].idxmax()
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Highest F1 Score",
            f"{best_row['F1 Score'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Corresponding Threshold",
            f"{best_row['Threshold']:.1f}"
        )

    st.info("""
    The documented threshold analysis shows that lower thresholds
    generally favor Recall, while increasing the threshold improves
    Precision. The threshold comparison helps evaluate the balance
    between false positives and false negatives.
    """)


# ============================================================
# PAGE 5 — NOISE ANALYSIS
# ============================================================

elif page == "🔊 Noise Analysis":

    st.title("🔊 Noise Sensitivity Analysis")

    st.write("""
    Week 4 focuses on evaluating model robustness under noisy
    industrial data conditions.

    Synthetic noise can be introduced into test data to study how
    model performance changes under realistic interference.
    """)

    st.subheader("🎯 Objective")

    st.write("""
    The objective is to understand how noise affects predictive
    maintenance classification and evaluate whether the model
    remains reliable under moderate data disturbances.
    """)

    st.markdown("---")

    st.subheader("🔬 Noise Validation Workflow")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        **Step 1**

        Start with the test dataset.

        **Step 2**

        Introduce controlled synthetic noise.

        **Step 3**

        Generate predictions using the trained model.
        """)

    with col2:

        st.markdown("""
        **Step 4**

        Calculate Precision, Recall and F1 Score.

        **Step 5**

        Compare performance under different noise levels.

        **Step 6**

        Evaluate model robustness.
        """)

    st.markdown("---")

    st.subheader("📌 Project Requirement")

    st.info("""
    The internship specification emphasizes model robustness against
    noisy industrial data and requires attention to Macro F1 under
    moderate noise conditions.
    """)

    st.warning("""
    The exact numerical noise-sensitivity results are not hardcoded
    here because the source project material does not provide the
    complete numerical noise-results table.
    """)

    st.success(
        "Noise sensitivity analysis is included as a core Week 4 "
        "project component."
    )


# ============================================================
# PAGE 6 — FINAL REPORT
# ============================================================

elif page == "📄 Final Report":

    st.title("📄 Final Project Report")

    st.header(
        "Contextual Predictive Maintenance"
    )

    st.write("""
    This project presents an end-to-end machine learning approach
    for predictive maintenance using contextual data fusion.
    """)

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write("""
    The objective is to develop a contextual predictive maintenance
    framework capable of identifying potential machine failures
    before breakdown occurs.
    """)

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    st.subheader("⚙️ Methodology")

    methodology = [
        "IoT Telemetry / Dataset Ingestion",
        "Data Exploration",
        "Signal Processing",
        "Contextual Data Fusion",
        "Feature Engineering",
        "Imbalanced Classification",
        "SMOTE",
        "LightGBM Classification",
        "5-Fold Stratified Cross Validation",
        "Model Evaluation",
        "Threshold Tuning",
        "Noise Sensitivity Analysis"
    ]

    for item in methodology:

        st.write(
            f"✔️ {item}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.subheader("📊 Documented Results")

    results_df = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Score": [
            "98.90%",
            "76.74%",
            "97.06%",
            "85.71%"
        ]
    })

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    st.subheader("🏁 Conclusion")

    st.write("""
    The final project evaluation provides a comprehensive view of
    predictive maintenance model performance.

    The analysis includes model performance comparison, threshold
    tuning, Precision-Recall analysis and evaluation of the balance
    between Precision and Recall.

    These analyses support model interpretation and demonstrate the
    usefulness of machine learning for predictive maintenance.
    """)

    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    st.subheader("👥 Project Team")

    st.write("""
    **Durga Anil — Team Lead**

    **Team Members**

    - Binit Binu
    - Krutika Thakur
    - Priyadharshini
    """)

    st.markdown("---")

    st.success(
        "🚀 Project Completed Successfully"
    )

    st.caption(
        "Data Science & Machine Learning Internship Project"
    )
