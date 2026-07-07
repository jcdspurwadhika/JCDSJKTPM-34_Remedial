import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bank Campaign Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── White background, red/yellow/black/green palette ─────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111111; }

    [data-testid="stSidebar"] {
        background-color: #1a5c1a;
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%; text-align: left;
        background: transparent; border: none;
        color: #FFFFFF !important; padding: 0.5rem 1rem;
        border-radius: 6px; font-size: 0.95rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #145214;
    }

    h1, h2, h3, h4 { color: #8B0000; }
    p, li, label { color: #111111; }

    .card-red {
        background: #fff5f5;
        border-left: 5px solid #CC0000;
        border-radius: 8px; padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .card-red h3 { color: #CC0000; margin: 0 0 0.3rem 0; font-size: 1.7rem; }
    .card-red p  { color: #555; margin: 0; font-size: 0.85rem; }

    .card-green {
        background: #f0fff0;
        border-left: 5px solid #1a5c1a;
        border-radius: 8px; padding: 1rem 1.2rem;
        margin: 0.4rem 0;
    }
    .card-green p { color: #111; margin: 0; }

    .card-yellow {
        background: #fffde7;
        border-left: 5px solid #DAA520;
        border-radius: 8px; padding: 1rem 1.2rem;
        margin: 0.4rem 0;
    }
    .card-yellow p { color: #111; margin: 0; }

    .section-divider {
        border: none; border-top: 2px solid #CC0000;
        margin: 1.5rem 0;
    }

    .stButton > button {
        background-color: #CC0000; color: #FFFFFF;
        border: none; border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #a00000; }

    .stMetric { background: #fff5f5; border-radius: 8px; padding: 0.5rem; }
    .stMetric label { color: #8B0000 !important; font-weight: 600; }
    .stMetric [data-testid="metric-container"] { color: #111 !important; }

    .stDataFrame { border: 1px solid #CC0000; border-radius: 6px; }
    .stExpander { border: 1px solid #CC0000 !important; border-radius: 8px; }
    [data-testid="stExpanderToggleIcon"] { color: #CC0000 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_FILE    = 'finalised_model.sav'
JOB_OPTS      = ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
MARITAL_OPTS  = ['divorced', 'married', 'single']
EDU_OPTS      = ['illiterate', 'basic 4 years', 'basic 6 years', 'basic 9 years',
                 'high school', 'professional course', 'university degree']
YES_NO_OPTS   = ['yes', 'no']
CONTACT_OPTS  = ['cellular', 'telephone']
MONTH_OPTS    = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
DAY_OPTS      = ['mon', 'tue', 'wed', 'thu', 'fri']
POUTCOME_OPTS = ['failure', 'nonexistent', 'success']
PHASE_OPTS    = ['pre_crisis', 'crisis', 'recovery']

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown(
        "<h2 style='color:#FFD700; margin-bottom:0.5rem;'>🏦 Navigation</h2>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    pages = {
        "🏠  Home":                  "Home",
        "🔍  Single Prediction":     "Single Prediction",
        "📂  Batch Prediction":      "Batch Prediction (CSV)",
        "🛠️  Train Model":           "Train Model",
        "📊  Model Documentation":   "Model Specs",
        "📖  Data Dictionary":       "Data Dictionary",
        "🌍  Macroeconomic Analysis":"Macroeconomic Analysis",
    }
    for label, key in pages.items():
        if st.sidebar.button(label, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#ccffcc;'>Alpha-Team · Purwadhika Data Science</small>",
        unsafe_allow_html=True
    )

# ── Home ──────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
        <h1 style='font-size:2.8rem; color:#8B0000; margin-bottom:0.3rem;'>
            🏦 Bank Campaign Marketing Predictor
        </h1>
        <p style='color:#555; font-size:1.1rem; margin-top:0;'>
            Predict Term Deposit Subscription with Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "66.5%",  "Recall — subscribers caught"),
        (c2, "81.3%",  "ROC-AUC"),
        (c3, "4.20×",  "More profit per flagged-customer vs. random"),
        (c4, "320%",   "Higher profit gain than baseline"),
    ]:
        with col:
            st.markdown(f"""
            <div class='card-red'>
                <h3>{val}</h3><p>{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Welcome to the Bank Marketing Campaign Predictor!

        This application uses **Machine Learning (LightGBM)** to predict whether a customer
        will subscribe to a term deposit based on their demographic information,
        previous campaign interactions, and macroeconomic indicators.

        #### What Can You Do Here?
        - **Single Prediction** — Predict subscription likelihood for individual customers
        - **Batch Prediction** — Upload CSV files for bulk predictions
        - **Train Model** — Train or retrain the model with your own data
        - **Model Documentation** — View model performance metrics and specifications
        - **Data Dictionary** — Understand all features used in the model
        - **Macroeconomic Analysis** — Learn about economic context during data collection

        #### Model Performance
        """)
        for line in [
            "✅ **Recall: 0.6655** — Target ≥0.60 met",
            "✅ **ROC-AUC: 0.8133** — Target ≥0.80 met",
            "⚠️ **PR-AUC: 0.4882** — Target ≥0.50 missed by 0.012",
        ]:
            color = "#1a5c1a" if "✅" in line else "#DAA520"
            st.markdown(
                f"<div class='card-{'green' if '✅' in line else 'yellow'}'>"
                f"<p>{line}</p></div>",
                unsafe_allow_html=True
            )

    with col2:
        st.info("""
        **Quick Facts**

        **Model:** LightGBM Classifier

        **Dataset:** 41,188 records

        **Features:** 20 input variables

        **Primary metric:** Recall
        (minimise missed subscribers)

        ---
        **Created by:**
        Tengku Arika Hazera

        *Purwadhika Data Science Final Project Remedial*
        """)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    nav_cards = [
        (c1, "🔍 Make Predictions",
         "Predict subscription probability for individual customers or upload CSV files for bulk predictions.",
         "Single Prediction", "Batch Prediction (CSV)"),
        (c2, "🛠️ Model Management",
         "Train or retrain the model with custom data, and view comprehensive model documentation.",
         "Train Model", "Model Specs"),
        (c3, "📖 Learn More",
         "Explore the data dictionary and understand macroeconomic factors affecting predictions.",
         "Data Dictionary", "Macroeconomic Analysis"),
    ]
    for col, title, desc, btn1_label, btn2_label in nav_cards:
        with col:
            st.markdown(f"""
            <div style='background:#fff5f5; border:1px solid #CC0000; border-radius:10px;
                        padding:1.2rem; min-height:130px; margin-bottom:0.5rem;'>
                <h4 style='color:#8B0000; margin:0 0 0.5rem 0;'>{title}</h4>
                <p style='color:#555; font-size:0.9rem; margin:0;'>{desc}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Go to {btn1_label.replace('Single ', '').replace(' (CSV)','')}",
                         key=f"home1_{btn1_label}", use_container_width=True):
                st.session_state.page = btn1_label
                st.rerun()
            if st.button(btn2_label.replace(" (CSV)", "").replace("Model ", ""),
                         key=f"home2_{btn2_label}", use_container_width=True):
                st.session_state.page = btn2_label
                st.rerun()

# ── Single Prediction ─────────────────────────────────────────────────────────
def page_single():
    st.header("🔍 Single Customer Prediction")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()

    model = load_model()
    if model is None:
        st.error(f"Model file `{MODEL_FILE}` not found. Place it in the same folder as app_white (1).py.")
        return

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal Info")
            age        = st.number_input("Age", 17, 98, 35, help="Range: 17–98")
            job        = st.selectbox("Job", JOB_OPTS,
                                       help="Students and retired customers convert best (~31% and ~25%)")
            marital    = st.selectbox("Marital Status", MARITAL_OPTS)
            education  = st.selectbox("Education", EDU_OPTS)
            housing    = st.selectbox("Has Housing Loan?", YES_NO_OPTS)
            loan       = st.selectbox("Has Personal Loan?", YES_NO_OPTS)
            is_default = st.selectbox("Is Default Status Known?", YES_NO_OPTS)

        with col2:
            st.subheader("Last Campaign")
            contact    = st.selectbox("Contact Type", CONTACT_OPTS,
                                       help="Cellular converts significantly better than telephone")
            month      = st.selectbox("Month", MONTH_OPTS,
                                       help="March and December show highest conversion rates")
            day        = st.selectbox("Day of Week", DAY_OPTS)
            was_before = st.selectbox("Was Contacted Before?", YES_NO_OPTS)
            campaign   = st.number_input("Campaign Contacts", 1, 15, 1,
                                          help="Range: 1–15; fewer contacts tend to convert better")
            previous   = st.number_input("Previous Contacts", 0, 7, 0)
            poutcome   = st.selectbox("Previous Outcome", POUTCOME_OPTS,
                                       help="'success' is the strongest positive signal")

        with col3:
            st.subheader("Macro Indicators")
            st.caption("These are the top model drivers — low rates + low employment → higher conversion")
            cons_idx    = st.number_input("Consumer Confidence Index",
                                           -50.8, -26.9, -40.0, step=0.1,
                                           help="Range: −50.8 to −26.9")
            euribor     = st.number_input("Euribor 3m Rate",
                                           0.634, 5.045, 1.0, step=0.001,
                                           help="Lower rates → higher subscription likelihood")
            nr_employed = st.number_input("Nr. Employed",
                                           4963.6, 5228.1, 5000.0, step=0.1,
                                           help="Lower = weaker economy = higher subscription rate")
            phase       = st.selectbox("Economic Phase", PHASE_OPTS,
                                        help="Recovery phase has the highest conversion rate (~44%)")
            pdays       = st.number_input("Days Since Last Contact",
                                           0, 27, 0,
                                           help="Leave 0 if never contacted before")

        submitted = st.form_submit_button("🔮 Predict Subscription",
                                           use_container_width=True)

    if submitted:
        pdays_val = np.nan if (pdays == 0 and was_before == 'no') else pdays
        input_df = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'housing': housing, 'loan': loan,
            'is_default_status_known': is_default,
            'contact': contact, 'month': month, 'day_of_week': day,
            'was_contacted_before': was_before,
            'campaign': campaign, 'previous': previous, 'poutcome': poutcome,
            'cons.conf.idx': cons_idx, 'euribor3m': euribor,
            'nr.employed': nr_employed, 'phase': phase, 'pdays': pdays_val
        }])

        try:
            proba      = model.predict_proba(input_df)[0][1]
            prediction = proba >= 0.5

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            r1, r2 = st.columns([1, 2])

            with r1:
                if prediction:
                    st.success("✅ **Likely to Subscribe**")
                    tier = ("🥇 Gold"   if proba >= 0.7 else
                            "🥈 Silver" if proba >= 0.5 else "🥉 Bronze")
                else:
                    st.error("❌ **Unlikely to Subscribe**")
                    tier = "🥉 Bronze"
                st.metric("Subscription Probability", f"{proba:.1%}")
                st.metric("Lead Tier", tier)
                st.caption("""
                **Lead tiers:**
                🥇 Gold ≥70% — top agents
                🥈 Silver 50–70% — standard call
                🥉 Bronze <50% — deprioritise
                """)

            with r2:
                bar_color = "#1a5c1a" if prediction else "#CC0000"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": bar_color},
                        "bgcolor": "#f9f9f9",
                        "steps": [
                            {"range": [0,  50], "color": "#ffe5e5"},
                            {"range": [50, 75], "color": "#fffde7"},
                            {"range": [75,100], "color": "#e8f5e9"},
                        ],
                        "threshold": {
                            "line": {"color": "#8B0000", "width": 3},
                            "thickness": 0.8, "value": 50,
                        },
                    },
                    title={"text": "Subscription Probability",
                           "font": {"color": "#8B0000"}}
                ))
                fig.update_layout(
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    height=280, margin=dict(t=60, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption("Check that all input columns match the model's expected features.")

# ── Batch Prediction ──────────────────────────────────────────────────────────
def page_batch():
    st.header("📂 Batch Prediction (CSV)")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()

    model = load_model()
    if model is None:
        st.error(f"Model file `{MODEL_FILE}` not found.")
        return

    st.markdown("""
    Upload a CSV with customer records. The model will add a **subscription probability**,
    **predicted outcome**, and **lead tier** for each row.
    """)

    required_cols = [
        'age','job','marital','education','housing','loan',
        'is_default_status_known','contact','month','day_of_week',
        'was_contacted_before','campaign','previous','poutcome',
        'cons.conf.idx','euribor3m','nr.employed','phase','pdays'
    ]
    with st.expander("Required column names"):
        st.code(", ".join(required_cols))

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:", data.head())

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return

        if st.button("Predict All"):
            try:
                data['subscription_probability'] = model.predict_proba(data)[:, 1]
                data['predicted_deposit']        = (
                    data['subscription_probability'] >= 0.5
                ).map({True: 'yes', False: 'no'})
                data['lead_tier'] = pd.cut(
                    data['subscription_probability'],
                    bins=[0, 0.5, 0.7, 1.0],
                    labels=['Bronze', 'Silver', 'Gold']
                )
                st.success("Prediction complete!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Subscribers",
                          int((data['predicted_deposit'] == 'yes').sum()))
                c2.metric("Avg. Probability",
                          f"{data['subscription_probability'].mean():.1%}")
                c3.metric("Gold Tier Leads",
                          int((data['lead_tier'] == 'Gold').sum()))

                st.dataframe(data)
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name='bank_predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error during prediction. Details: {e}")

# ── Train Model ───────────────────────────────────────────────────────────────
def page_train():
    trained_model = 'bank_model_trained.sav'
    st.header("🛠️ Train / Retrain Model")
    st.info(f"This will train a new LightGBM model and save it as `{trained_model}`.")
    st.warning("Ensure your CSV has a `deposit` column (yes/no) as the target.")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()

    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    if train_file:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import (RobustScaler, OneHotEncoder,
                                            OrdinalEncoder, TargetEncoder)
        from sklearn.impute import SimpleImputer
        from lightgbm import LGBMClassifier
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, fbeta_score)

        df = pd.read_csv(train_file)
        st.write(f"Training data shape: {df.shape}")

        if st.button("Start Training"):
            with st.spinner("Training model..."):
                X = df.drop(columns=['deposit'])
                y = df['deposit'].map({'yes': 1, 'no': 0})

                numerical_features  = ['age','campaign','previous','pdays',
                                        'cons.conf.idx','euribor3m','nr.employed']
                ordinal_features    = ['month','day_of_week','education']
                target_features     = ['job']
                categorical_features= [c for c in X.select_dtypes('object').columns
                                        if c not in ordinal_features + target_features]

                month_order     = ['jan','feb','mar','apr','may','jun',
                                   'jul','aug','sep','oct','nov','dec']
                day_order       = ['mon','tue','wed','thu','fri']
                education_order = ['illiterate','basic 4 years','basic 6 years',
                                   'basic 9 years','high school',
                                   'professional course','university degree']

                preprocessor = ColumnTransformer([
                    ('num', Pipeline([
                        ('scaler', RobustScaler()),
                        ('imputer', SimpleImputer(strategy='median'))
                    ]), numerical_features),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(drop='first',
                                                   sparse_output=False,
                                                   handle_unknown='ignore'))
                    ]), categorical_features),
                    ('ord', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OrdinalEncoder(
                            categories=[month_order, day_order, education_order],
                            handle_unknown='use_encoded_value', unknown_value=-1))
                    ]), ordinal_features),
                    ('target', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', TargetEncoder(random_state=42))
                    ]), target_features),
                ])

                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LGBMClassifier(
                        class_weight='balanced',
                        learning_rate=0.02,
                        max_depth=4,
                        n_estimators=300,
                        random_state=42,
                        verbose=-1
                    ))
                ])
                pipeline.fit(X, y)
                joblib.dump(pipeline, trained_model)
                st.success(f"Model trained and saved as `{trained_model}`.")

                y_pred = pipeline.predict(X)
                prc = precision_score(y, y_pred)
                rec = recall_score(y, y_pred)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{accuracy_score(y, y_pred):.4f}")
                c2.metric("Precision", f"{prc:.4f}")
                c3.metric("Recall",    f"{rec:.4f}")
                c4.metric("F2-Score",
                          f"{fbeta_score(y, y_pred, beta=2):.4f}")

# ── Model Documentation ───────────────────────────────────────────────────────
def page_model_specs():
    st.header("📊 Model Documentation")
    st.markdown("*Created by Fatimah Azzahra, Tengku Arika Hazera, Yonathan Hary Hutagalung — Purwadhika Data Science Final Project*")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_1, col_2, col_3 = st.columns(3)
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()

    with col_1:
        st.subheader("Performance Metrics")
        for name, val, note, good in [
            ("Recall (primary)", "0.6655", "Target ≥0.60 ✅", True),
            ("F2-Score",         "0.5706", "Target ≥0.55 ✅", True),
            ("ROC-AUC",          "0.8133", "Target ≥0.80 ✅", True),
            ("PR-AUC",           "0.4882", "Target ≥0.50 ⚠️", False),
            ("Precision",        "0.3632", "", True),
            ("F1-Score",         "0.4700", "", True),
        ]:
            st.metric(name, val)
            if note:
                color = "#1a5c1a" if good else "#DAA520"
                st.markdown(
                    f"<small style='color:{color}'>{note}</small>",
                    unsafe_allow_html=True
                )

    with col_2:
        st.subheader("Configuration")
        st.code("""
classifier__class_weight:  balanced
classifier__learning_rate: 0.02
classifier__n_estimators:  300
classifier__max_depth:     4
        """, language="yaml")

        st.subheader("Pipeline")
        st.markdown("""
        * **Numerical:** RobustScaler + SimpleImputer (median)
        * **Categorical:** SimpleImputer + OneHotEncoder
        * **Ordinal:** OrdinalEncoder (month, day_of_week, education)
        * **Target:** TargetEncoder (job)
        """)

        st.subheader("Before vs. After Tuning")
        comp = pd.DataFrame({
            "Metric":  ["Recall", "Precision", "FN (missed)", "TP (caught)", "Net Profit"],
            "Before":  ["0.2685", "0.6349", "654", "240", "EUR 10,871"],
            "After":   ["0.6655", "0.3632", "299", "595", "EUR 25,803"],
        })
        st.dataframe(comp, hide_index=True, use_container_width=True)

    with col_3:
        st.subheader("Dataset Information")
        st.write("**Dataset:** Bank Campaign Marketing Dataset")
        st.write("**Source:** UCI Machine Learning Repository")
        st.write("**Size:** 41,188 rows")
        st.write("**Features:** 20 input variables + 1 target (deposit)")

        st.subheader("SHAP — Top Drivers")
        shap_df = pd.DataFrame({
            "Rank":      [1, 2, 3, 4, 5],
            "Feature":   ["nr.employed", "cons.conf.idx", "contact_telephone",
                          "euribor3m", "age"],
            "Direction": ["Low → ↑ subscribe", "Low → ↑ subscribe",
                          "Landline → ↓", "Low → ↑ subscribe",
                          "U-shape (young & retired)"],
        })
        st.dataframe(shap_df, hide_index=True, use_container_width=True)

# ── Data Dictionary ───────────────────────────────────────────────────────────
def page_data_dict():
    st.header("📖 Data Dictionary")
    st.write("Reference guide for all features used in the bank marketing campaign model.")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()

    with st.expander("1. Bank Client Data", expanded=True):
        st.markdown("""
        * **Age:** Numerical age of the client (17–98).
        * **Job:** Type of job — *student* and *retired* show the highest conversion rates.
        * **Marital:** Marital status (*divorced* also includes widowed).
        * **Education:** Level of education — higher education correlates with higher conversion.
        * **Is default status known:** Whether we have information about the customer's default status.
        * **Has a housing loan?:** Whether the customer has a housing loan.
        * **Has a personal loan?:** Whether the customer has a personal loan.
        """)

    with st.expander("2. Last Contact Information"):
        st.markdown("""
        * **Contact:** Contact channel — *cellular* converts ~3× better than *telephone*.
        * **Month:** Last contact month — March and December show the highest subscription rates.
        * **Day of week:** Last contact day of the week.
        * **Campaign:** Number of contacts during this campaign (1–15) — fewer is better.
        * **Pdays:** Days since last contact; replaced with NaN when customer was never contacted.
        * **Previous:** Number of contacts before this campaign.
        * **Poutcome:** Previous campaign outcome — *success* is the strongest positive signal.
        * **Was contacted before?:** Binary flag — was the customer contacted in any prior campaign.
        """)

    with st.expander("3. Macroeconomic Indicators (Top SHAP Drivers)"):
        st.markdown("""
        * **Euribor 3m:** Euribor 3-month rate — **lower rate → higher subscription likelihood**.
        * **Nr. Employed:** Quarterly number of employees — **lower → higher subscription**.
        * **Cons. Conf. Index:** Consumer confidence — **lower confidence → higher subscription**.
        * **Phase:** Economic phase (*pre_crisis*, *crisis*, *recovery*) — recovery ~44% conversion.
        """)
        st.warning("""
        `emp.var.rate` and `cons.price.idx` were **dropped** due to multicollinearity
        (ρ > 0.91 with retained features). `duration` was dropped due to data leakage
        (only known after the call ends).
        """)

# ── Macroeconomic Analysis ────────────────────────────────────────────────────
def page_macro():
    st.header("🌍 Macroeconomic Context & Indicators")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"; st.rerun()
    st.info("The model utilizes data collected during the 2008–2010 financial crisis period.")

    st.markdown("""
    The dataset was collected during the 2008 financial crisis, characterized by high interest rates,
    low customer confidence, and a growing unemployment rate. The data shows the **opposite** of
    what one might expect: lower rates and weaker economic conditions are associated with *more*
    term deposit subscriptions — customers seek safety during uncertainty.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Monetary Drivers")
        st.markdown("""
        <div class='card-green'>
            <p><strong>Euribor 3m:</strong> Measures interbank interest rates.
            Lower rates signal economic stress and drive customers toward the safety of term deposits.</p>
        </div>""", unsafe_allow_html=True)

        st.subheader("2. Economic Health")
        st.markdown("""
        <div class='card-green'>
            <p><strong>Nr. Employed:</strong> Measures job market conditions.
            Fewer people employed → weaker economy → customers seek safe savings vehicles.</p>
        </div>""", unsafe_allow_html=True)

        st.subheader("3. Psychological Factors")
        st.markdown("""
        <div class='card-yellow'>
            <p><strong>Cons. Conf. Index:</strong> Measures how confident customers are about
            spending or saving. Low confidence → customers prefer safe, predictable products.</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Subscription Rate by Economic Phase")
        phase_df = pd.DataFrame({
            "Phase":   ["Pre-Crisis", "Crisis", "Recovery"],
            "Rate %":  [4, 12, 44],
            "Color":   ["#CC0000", "#DAA520", "#1a5c1a"]
        })
        fig = go.Figure(go.Bar(
            x=phase_df["Phase"], y=phase_df["Rate %"],
            marker_color=phase_df["Color"],
            text=[f"{v}%" for v in phase_df["Rate %"]],
            textposition="auto"
        ))
        fig.update_layout(
            title="Conversion Rate by Economic Phase",
            yaxis_title="Subscription Rate (%)",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font_color="#111111", height=320
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("The Crisis Timeline (2008–2010)")
        st.warning("""
        **The Chain Reaction:**

        1. **Pre-Crisis (2008):** EURIBOR spikes → banks desperate for cash.
           Customers have high confidence, low motivation to save.

        2. **Crisis (mid 2008–2009):** Confidence collapses, employment drops.
           Subscription rates begin to rise as uncertainty grows.

        3. **Recovery (2009–2010):** EURIBOR falls to ~0.6–1.0%.
           **Subscription rate peaks at ~44%** — customers seek deposit safety.
        """)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    sidebar()
    page = st.session_state.page

    if   page == "Home":                  page_home()
    elif page == "Single Prediction":     page_single()
    elif page == "Batch Prediction (CSV)":page_batch()
    elif page == "Train Model":           page_train()
    elif page == "Model Specs":           page_model_specs()
    elif page == "Data Dictionary":       page_data_dict()
    elif page == "Macroeconomic Analysis":page_macro()

if __name__ == "__main__":
    main()
