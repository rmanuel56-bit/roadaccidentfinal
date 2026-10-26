
"""
RACC Road Accident Analysis Dashboard
Run:
    python -m streamlit run racc_streamlit_app.py

Keep these files in the same folder:
    racc_streamlit_app.py
    style.css
    part_2_data.py
    part_3_models.py
    part_4_results.py
    requirements.txt
"""

from pathlib import Path
from datetime import date, time
import hashlib

import pandas as pd
import streamlit as st

try:
    from part_2_data import (
        REQUIRED,
        prepare_data,
        make_counts,
        statistical_analysis,
        features,
    )
    from part_3_models import (
        classification_analysis,
        prediction_analysis,
    )
    from part_4_results import (
        forecasting_analysis,
        download_package,
    )

except ModuleNotFoundError as exc:
    st.error(f"Missing file or Python package: {exc.name}")
    st.info(
        "Keep all three part_*.py files in the same folder "
        "and install requirements.txt."
    )
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RACC | Road Safety Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

def load_css():
    css_file = Path(__file__).resolve().parent / "style.css"

    if css_file.is_file():
        css_content = css_file.read_text(encoding="utf-8")
        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "style.css was not found. Place style.css "
            "in the same folder as racc_streamlit_app.py."
        )


# ============================================================
# DASHBOARD HEADER
# ============================================================

def show_header():
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="header-icon">🚦</div>
            <div class="header-content">
                <div class="header-eyebrow">
                    ROAD SAFETY ANALYTICS SYSTEM
                </div>
                <h1>RACC Dashboard</h1>
                <p>
                    Road Accident Patterns in Bunawan,
                    Agusan del Sur
                </p>
                <div class="header-tags">
                    <span>Statistical Analysis</span>
                    <span>Machine Learning</span>
                    <span>Forecasting</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CENTERED TOP NAVIGATION
# ============================================================

def show_navigation():
    nav_options = [
        "1. Upload and Cleaning",
        "2. Statistics and EDA",
        "3. Classification and Prediction",
        "4. Forecasting and Downloads",
    ]

    if "analysis_page" not in st.session_state:
        st.session_state["analysis_page"] = nav_options[0]

    page = st.radio(
        "Dashboard Navigation",
        options=nav_options,
        horizontal=True,
        key="analysis_page",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="navigation-divider"></div>',
        unsafe_allow_html=True,
    )

    return page


# ============================================================
# TABLE DISPLAY AND CSV DOWNLOAD
# ============================================================

def show_table(title, frame, key):
    st.subheader(title)

    if frame is None or frame.empty:
        st.info("No rows meet the requirements for this table.")
        return

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=key + ".csv",
        mime="text/csv",
        key="download_" + key,
    )


# ============================================================
# RUN ANALYSIS SAFELY
# ============================================================

def run_analysis(bundle, key, function, *args):
    try:
        with st.spinner(f"Running {key.replace('_', ' ')}..."):
            bundle["results"][key] = function(*args)

        st.success("Analysis completed successfully.")

    except Exception as exc:
        st.error(f"{key.replace('_', ' ').capitalize()}: {exc}")

        with st.expander("View error details"):
            st.exception(exc)


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

def show_classification(result):
    st.markdown("### Selected classification model")
    st.info(f"Selected using development data: {result['selected_name']}")

    show_table(
        "Classification Performance",
        result["metrics"],
        "classification_metrics",
    )

    if (
        not result["metrics"].empty
        and "ROC_AUC" in result["metrics"].columns
        and result["metrics"].iloc[0]["ROC_AUC"] < 0.55
    ):
        st.warning(
            "Held-out discrimination is close to chance. "
            "Feature rankings are exploratory, not validated risk factors."
        )

    sections = [
        ("Development Comparison", "cv"),
        ("Train/Test Dates", "split"),
        ("Class-Level Results", "report"),
        ("Four-Class Reference Results", "multiclass_report"),
    ]

    for title, key in sections:
        with st.expander(title):
            show_table(
                title,
                result[key],
                "classification_" + key,
            )

    st.markdown("### Confusion Matrix")
    st.caption(
        "Binary classification: 0 = minor/no injury; "
        "1 = major/fatal."
    )
    st.dataframe(result["confusion"], width="stretch")

    st.markdown("### Variable Importance")

    importance = result["importance"]

    if not importance.empty:
        st.bar_chart(
            importance.set_index("variable")["macro_F1_drop"]
        )

    show_table(
        "Variable Importance Details",
        importance,
        "classification_importance",
    )

    show_table(
        "Variable-Set Comparisons",
        result["variable_combinations"],
        "classification_variable_sets",
    )

    show_table(
        "Observed Weather/Road/Vehicle Combinations",
        result["observed_combinations"],
        "observed_combinations",
    )

    st.caption(
        "Combinations use predictions from earlier-year training folds. "
        "Importance is post-evaluation and does not establish causation."
    )

    with st.expander("Simple Reference-Tree Rules"):
        st.code(result["rules"], language="text")


# ============================================================
# SEVERITY PREDICTION FORM
# ============================================================

def prediction_form(bundle):
    result = bundle["results"].get("classification")

    if result is None:
        return

    st.markdown("### Try a Severity Prediction")
    st.caption(
        "Enter an example record to obtain the model's estimated "
        "probability of major injury or fatality."
    )

    clean = bundle["data"]["clean"]

    weather_options = sorted(clean["weather"].dropna().unique().tolist())
    road_options = sorted(
        clean["road_condition"].dropna().unique().tolist()
    )

    if not weather_options or not road_options:
        st.warning(
            "Prediction is unavailable because weather or road-condition "
            "options are missing from the cleaned dataset."
        )
        return

    with st.form("severity_form"):
        a, b = st.columns(2)

        when = a.date_input("Date", date(2024, 8, 15))
        clock = b.time_input("Time", time(20, 0))

        location = a.text_input("Location", "P1 San Teodoro")
        vehicle = b.text_input("Vehicle Type", "Motorcycle")

        weather = a.selectbox("Weather", weather_options)
        road = b.selectbox("Road Condition", road_options)

        gender = a.selectbox(
            "Gender as recorded",
            ["Male", "Female", "Male & Female"],
        )

        submitted = st.form_submit_button(
            "Predict Severity",
            use_container_width=True,
        )

    if submitted:
        try:
            record = dict(
                zip(
                    REQUIRED,
                    [
                        when.strftime("%d/%m/%Y"),
                        clock.strftime("%I:%M %p"),
                        location,
                        weather,
                        road,
                        vehicle,
                        "",
                        "",
                        gender,
                    ],
                )
            )

            incoming = prepare_data(
                pd.DataFrame([record]).to_csv(index=False).encode("utf-8")
            )["clean"]

            probability = result["model"].predict_proba(
                features(incoming)
            )[0, 1]

            st.metric(
                "Estimated Probability of Major Injury or Fatality",
                f"{probability:.1%}",
            )

            st.caption(
                "Illustrative prediction among reported records. "
                "This is not an individual accident forecast or a "
                "validated measure of real-world risk. Review the "
                "model performance before interpreting this result."
            )

        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")


# ============================================================
# MAIN DASHBOARD
# ============================================================

def main():
    # Load CSS before rendering dashboard components
    load_css()

    # Display dashboard header
    show_header()

    # Navigation now appears below the header in the MAIN PAGE
    page = show_navigation()

    # --------------------------------------------------------
    # SIDEBAR BRANDING AND DATASET SETTINGS
    # --------------------------------------------------------

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🚦</div>
            <div class="sidebar-title">RACC Dashboard</div>
            <div class="sidebar-subtitle">
                Road Accident Analysis System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Dataset")

    upload = st.sidebar.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        key="dataset_upload",
    )

    with st.sidebar.expander("CSV and Cleaning Settings"):
        fix = st.checkbox(
            "Apply documented provisional date corrections",
            True,
            key="date_fixes",
        )

        encoding = st.selectbox(
            "Encoding",
            ["utf-8-sig", "utf-8", "cp1252", "latin-1"],
            key="encoding",
        )

        separator_name = st.selectbox(
            "Separator",
            ["Comma", "Semicolon", "Tab"],
            key="separator",
        )

        separator = {
            "Comma": ",",
            "Semicolon": ";",
            "Tab": "\t",
        }[separator_name]

    coverage = st.sidebar.checkbox(
        "Assume complete recording calendar",
        True,
        key="calendar_coverage",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "RACC | Road Safety Analytics\n\n"
        "Bunawan, Agusan del Sur"
    )

    # --------------------------------------------------------
    # REQUIRE DATASET UPLOAD
    # --------------------------------------------------------

    if upload is None:
        st.session_state.pop("racc_bundle", None)

        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-icon">📂</div>
                <h2>Welcome to RACC Analytics</h2>
                <p>
                    Upload your road accident CSV dataset using the
                    sidebar to begin cleaning, statistical analysis,
                    classification, prediction and forecasting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Required CSV column order"):
            st.code(",".join(REQUIRED), language="text")

        st.info("Upload your RACC.csv dataset using the left sidebar.")
        return

    # --------------------------------------------------------
    # PREPARE AND CACHE DATA IN SESSION
    # --------------------------------------------------------

    signature = hashlib.sha256(
        upload.getvalue()
        + str((fix, encoding, separator, coverage)).encode()
    ).hexdigest()

    bundle = st.session_state.get("racc_bundle")

    if bundle is None or bundle["signature"] != signature:
        st.session_state.pop("racc_bundle", None)

        try:
            prepared = prepare_data(
                upload.getvalue(),
                fix,
                encoding,
                separator,
            )

        except Exception as exc:
            st.error(f"Could not read this dataset: {exc}")
            return

        bundle = {
            "signature": signature,
            "data": prepared,
            "results": {},
            "filename": upload.name,
        }

        st.session_state["racc_bundle"] = bundle

    d = bundle["data"]["clean"]
    results = bundle["results"]

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.markdown("### Dataset Overview")

    a, b, c = st.columns(3)

    a.metric(
        "Total Reported Records",
        f"{len(d):,}",
        help="Number of rows in the cleaned dataset.",
    )

    b.metric(
        "Valid Injury Grades",
        f"{int(d.injury_grade.notna().sum()):,}",
        help="Records with a usable injury-grade value.",
    )

    c.metric(
        "Completed Analyses",
        f"{len(results):,}",
        help="Number of analysis tasks completed in this session.",
    )

    st.caption(
        "Repeated rows are retained. Counts describe reported records, "
        "not verified unique accidents or accident rates."
    )

    st.markdown("---")

    # ========================================================
    # PAGE 1: UPLOAD AND CLEANING
    # ========================================================

    if page.startswith("1."):
        st.header("Upload and Data Cleaning")

        show_table(
            "Data Quality Audit",
            bundle["data"]["audit"],
            "data_quality",
        )

        st.markdown("### Duplicate and Missing Data Checks")

        x, y, z = st.columns(3)

        x.metric(
            "Exact Repeats",
            f"{bundle['data']['exact_repeats']:,}",
        )

        y.metric(
            "Unusable Dates",
            f"{int(d.date.isna().sum()):,}",
        )

        z.metric(
            "Missing/Unusable Times",
            f"{int(d.hour.isna().sum()):,}",
        )

        show_table(
            "Dates Requiring Review",
            bundle["data"]["date_review"],
            "date_review",
        )

        show_table(
            "Cleaned Dataset",
            d,
            "cleaned_data",
        )

        st.info(
            "Proposed year/date repairs require source-owner confirmation. "
            "Conscious and Unconscious are not injury grades. "
            "Missing target labels are not imputed."
        )

    # ========================================================
    # PAGE 2: STATISTICS AND EDA
    # ========================================================

    elif page.startswith("2."):
        st.header("Statistical Analysis and Exploratory Data Analysis")

        st.write(
            "Explore accident-record frequencies, associations, "
            "data quality, monthly reported records and severity labels."
        )

        if st.button(
            "Run Statistical Analysis",
            key="run_statistics",
            type="primary",
        ):
            run_analysis(
                bundle,
                "statistics",
                statistical_analysis,
                d,
            )

        if "statistics" in results:
            stats = results["statistics"]

            sections = [
                ("Category Frequencies", "frequencies"),
                ("Exploratory Association Tests", "associations"),
                ("Severity-Label Coverage by Year", "severity_coverage"),
                ("Numeric Description", "numeric_summary"),
            ]

            for title, key in sections:
                show_table(
                    title,
                    stats[key],
                    "statistics_" + key,
                )

            st.caption(
                "Association tests collapse provisional incidents, "
                "permute labels within year and adjust p-values using "
                "Benjamini–Hochberg. Independence and reporting "
                "completeness remain assumptions."
            )

        st.markdown("### Explore Accident Categories")

        category = st.selectbox(
            "Select a variable",
            [
                "weather",
                "road_condition",
                "vehicle",
                "area",
                "severity",
                "gender_group",
                "cause",
            ],
            key="eda_variable",
        )

        st.bar_chart(
            d[category]
            .fillna("missing")
            .value_counts()
            .head(20)
        )

        st.markdown("### Missing Values")

        st.bar_chart(
            d[
                ["date", "hour", "location", "injury_grade"]
            ].isna().mean().mul(100)
        )

        dated = d.loc[d.date.notna()]

        if not dated.empty:
            st.markdown("### Monthly Reported Rows")

            monthly = (
                dated.groupby(
                    pd.Grouper(key="date", freq="MS")
                )
                .size()
            )

            st.line_chart(monthly)

            st.markdown("### Severity Vocabulary by Year")

            st.bar_chart(
                pd.crosstab(
                    d.year,
                    d.severity.fillna("missing"),
                )
            )

    # ========================================================
    # PAGE 3: CLASSIFICATION AND PREDICTION
    # ========================================================

    elif page.startswith("3."):
        st.header("Classification and Numeric Prediction")

        st.write(
            "Severity classification uses injury-grade recording years "
            "with at least 50% valid grades. The latest eligible dates "
            "are held out; preprocessing is fitted within training folds."
        )

        if st.button(
            "Run Classification",
            key="run_classification",
            type="primary",
        ):
            run_analysis(
                bundle,
                "classification",
                classification_analysis,
                d,
            )

        if "classification" in results:
            show_classification(results["classification"])
            prediction_form(bundle)

        st.markdown("---")
        st.subheader("Predict Daily Reported-Row Counts")

        st.caption(
            "One-day-ahead predictions use observed earlier days. "
            "A calendar day absent from the file is assumed to have "
            "zero recorded rows. MAE selection can favor a zero-median "
            "baseline; compare all error metrics."
        )

        if not coverage:
            st.info(
                "Enable the complete recording-calendar assumption "
                "to model counts."
            )

        if st.button(
            "Run Numeric Prediction",
            disabled=not coverage,
            key="run_prediction",
            type="primary",
        ):
            try:
                counts = make_counts(d)

            except Exception as exc:
                st.error(str(exc))

            else:
                run_analysis(
                    bundle,
                    "prediction",
                    prediction_analysis,
                    counts,
                )

        if "prediction" in results:
            r = results["prediction"]

            show_table(
                "Numeric Prediction Performance",
                r["metrics"],
                "prediction_metrics",
            )

            st.markdown("### Actual vs Predicted Daily Counts")

            st.line_chart(
                r["predictions"].set_index("date")[
                    ["actual", "predicted"]
                ]
            )

            show_table(
                "Next Day After the Dataset End",
                r["next_day"],
                "next_day",
            )

            show_table(
                "Daily Prediction Variables",
                r["importance"],
                "prediction_importance",
            )

            show_table(
                "Daily Variable-Set Comparisons",
                r["variable_combinations"],
                "prediction_variable_sets",
            )

            with st.expander("Development Comparison and Split"):
                show_table(
                    "Development MAE",
                    r["cv"],
                    "prediction_cv",
                )

                st.dataframe(r["split"], width="stretch")

            st.caption(
                "Importance describes the best learned reference if "
                "a constant baseline wins; it does not show that the "
                "learned model improves on that baseline."
            )

    # ========================================================
    # PAGE 4: FORECASTING AND DOWNLOADS
    # ========================================================

    else:
        st.header("Forecasting, Model Performance and Downloads")

        horizon = st.slider(
            "Months Beyond the Dataset End",
            min_value=1,
            max_value=24,
            value=12,
            key="forecast_horizon",
        )

        if not coverage:
            st.info(
                "Enable the complete recording-calendar assumption "
                "to model counts."
            )

        if st.button(
            "Run Monthly Forecasting",
            disabled=not coverage,
            key="run_forecast",
            type="primary",
        ):
            try:
                counts = make_counts(d)

            except Exception as exc:
                st.error(str(exc))

            else:
                run_analysis(
                    bundle,
                    "forecast",
                    forecasting_analysis,
                    counts,
                    horizon,
                )

        if "forecast" in results:
            r = results["forecast"]

            show_table(
                "Monthly Forecast Performance",
                r["metrics"],
                "forecast_metrics",
            )

            st.markdown("### Actual vs Predicted Monthly Counts")

            st.line_chart(
                r["holdout"].set_index("date")[
                    ["actual", "predicted"]
                ]
            )

            show_table(
                "Forecast After the Dataset End",
                r["future"],
                "future_forecast",
            )

            st.markdown("### Forecast and Approximate 90% Interval")

            st.line_chart(
                r["future"].set_index("date")[
                    [
                        "predicted",
                        "approx_90pct_low",
                        "approx_90pct_high",
                    ]
                ]
            )

            st.caption(
                f"Approximate 90% intervals use "
                f"{r['calibration_errors']} development errors. "
                f"Holdout coverage: {r['interval_coverage']:.1%}. "
                "Temporal dependence and small samples limit "
                "coverage guarantees."
            )

            st.info(
                "Forecast dates follow the last dataset month and may "
                "already be in the past. Update the data before current "
                "operational use."
            )

            with st.expander("Forecast Development Comparison"):
                show_table(
                    "Development MAE",
                    r["cv"],
                    "forecast_cv",
                )

        performance = [
            r["metrics"].assign(task=task)
            for task, r in results.items()
            if "metrics" in r
        ]

        if performance:
            show_table(
                "All Completed Model Evaluations",
                pd.concat(performance, ignore_index=True),
                "all_model_metrics",
            )

        st.markdown("### Export Your Results")

        st.caption(
            "Download the cleaned data and the analyses completed "
            "during this session."
        )

        st.download_button(
            "Download All Completed Results (ZIP)",
            download_package(
                bundle["data"],
                results,
                upload.name,
            ),
            file_name="RACC_results.zip",
            mime="application/zip",
            key="download_results",
            type="primary",
        )

        st.caption(
            "Run statistics, classification, numeric prediction and "
            "forecasting to include every analysis in the ZIP."
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dashboard-footer">
            <span>RACC Road Safety Analytics</span>
            <span>Bunawan, Agusan del Sur</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
