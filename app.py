import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161820;
    border-right: 1px solid #2a2d3a;
}
[data-testid="stSidebar"] * {
    color: #e8e4dc !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Metric cards */
.metric-card {
    background: #161820;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #c4874a; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #c4874a;
}
.metric-sub {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* Risk gauge */
.risk-container {
    background: #161820;
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.risk-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1rem;
}
.risk-score {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.risk-verdict {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    display: inline-block;
    margin-top: 0.5rem;
}
.risk-low    { color: #4ade80; border: 1px solid #4ade80; }
.risk-medium { color: #facc15; border: 1px solid #facc15; }
.risk-high   { color: #f97316; border: 1px solid #f97316; }
.risk-critical { color: #ef4444; border: 1px solid #ef4444; }

/* Section divider */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e4dc;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
}

/* Model comparison table */
.model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.model-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid #2a2d3a;
}
.model-table td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #1e2030;
    color: #e8e4dc;
}
.model-table tr:hover td { background: #1e2030; }
.model-table .winner td { color: #c4874a; font-weight: 600; }
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    background: rgba(196,135,74,0.15);
    color: #c4874a;
    border: 1px solid rgba(196,135,74,0.4);
    margin-left: 0.5rem;
}

/* Inputs */
.stSlider > div > div { background: #2a2d3a !important; }
.stSelectbox > div > div { background: #161820 !important; border: 1px solid #2a2d3a !important; }
.stRadio > div { gap: 0.5rem; }

/* Button */
.stButton > button {
    background: #c4874a !important;
    color: #0f1117 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: #d4974a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(196,135,74,0.3) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161820;
    border-radius: 8px;
    gap: 0;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    background: transparent;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #2a2d3a !important;
    color: #e8e4dc !important;
}

/* Info boxes */
.info-box {
    background: rgba(196,135,74,0.08);
    border: 1px solid rgba(196,135,74,0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #c4a87a;
    margin: 1rem 0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model files ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('model/model.pkl',         'rb') as f: model         = pickle.load(f)
    with open('model/all_models.pkl',    'rb') as f: all_models    = pickle.load(f)
    with open('model/scaler.pkl',        'rb') as f: scaler        = pickle.load(f)
    with open('model/feature_names.pkl', 'rb') as f: feature_names = pickle.load(f)
    with open('model/metadata.pkl',      'rb') as f: metadata      = pickle.load(f)
    with open('model/threshold.pkl',     'rb') as f: threshold     = pickle.load(f)
    return model, all_models, scaler, feature_names, metadata, threshold

model, all_models, scaler, feature_names, metadata, threshold = load_models()


# ── Feature engineering (must match training exactly) ─────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    df['TotalCharges']         = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['AvgMonthlySpend']      = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargePerMonth_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['SpendTrend']           = df['MonthlyCharges'] - df['AvgMonthlySpend']
    df['TenureBucket']         = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                                        labels=[0,1,2,3], include_lowest=True).astype(int)

    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    df['NumServices']          = df[service_cols].apply(lambda r: sum(v==1 for v in r), axis=1)
    df['HasInternet']          = (df['InternetService'] != 'No').astype(int)
    df['IsMonthToMonth']       = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsLongTermContract']   = (df['Contract'] == 'Two year').astype(int)
    df['HasNoSupport']         = ((df['TechSupport']==0) & (df['OnlineSecurity']==0)).astype(int)
    df['SeniorAlone']          = ((df['SeniorCitizen']==1) & (df['Partner']==0)).astype(int)
    df['HighSpendNewCustomer'] = ((df['MonthlyCharges']>70) & (df['tenure']<12)).astype(int)
    df['LowEngagement']        = ((df['NumServices']<=1) & (df['IsMonthToMonth']==1)).astype(int)

    # One-hot encode Contract, InternetService, PaymentMethod
    df = pd.get_dummies(df, columns=['Contract','InternetService','PaymentMethod'], drop_first=True)

    # Align to training feature set
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


# ── Sidebar — customer inputs ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#e8e4dc;
                    letter-spacing:-0.02em;'>📡 Churn<br><span style='color:#c4874a'>Intelligence</span></div>
        <div style='font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.15em;
                    text-transform:uppercase;color:#6b7280;margin-top:0.4rem;'>
            Telco Customer Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Account Info</div>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0,
                                       value=float(monthly_charges * tenure), step=10.0)
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    gender          = st.selectbox("Gender", ["Male", "Female"])
    partner         = st.selectbox("Has Partner", ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Services</div>',
                unsafe_allow_html=True)
    phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot      = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Billing</div>',
                unsafe_allow_html=True)
    contract         = st.selectbox("Contract Type",
                                    ["Month-to-month", "One year", "Two year"])
    paperless_billing= st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method   = st.selectbox("Payment Method",
                                    ["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyse Customer")


# ── Build input dict ───────────────────────────────────────────────────────────
def yn(val): return 1 if val == "Yes" else 0

raw_input = {
    'tenure':           tenure,
    'MonthlyCharges':   monthly_charges,
    'TotalCharges':     total_charges,
    'SeniorCitizen':    yn(senior_citizen),
    'gender':           1 if gender == "Male" else 0,
    'Partner':          yn(partner),
    'Dependents':       yn(dependents),
    'PhoneService':     yn(phone_service),
    'MultipleLines':    1 if multiple_lines == "Yes" else 0,
    'InternetService':  internet_service,
    'OnlineSecurity':   1 if online_security == "Yes" else 0,
    'OnlineBackup':     1 if online_backup == "Yes" else 0,
    'DeviceProtection': 1 if device_prot == "Yes" else 0,
    'TechSupport':      1 if tech_support == "Yes" else 0,
    'StreamingTV':      1 if streaming_tv == "Yes" else 0,
    'StreamingMovies':  1 if streaming_movies == "Yes" else 0,
    'Contract':         contract,
    'PaperlessBilling': yn(paperless_billing),
    'PaymentMethod':    payment_method,
}


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2rem'>
    <h1 style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
               color:#e8e4dc;letter-spacing:-0.03em;margin-bottom:0.3rem;'>
        Customer Churn Prediction
    </h1>
    <p style='color:#6b7280;font-size:0.9rem;margin:0;'>
        Configure a customer profile in the sidebar and click <strong style="color:#c4874a">Analyse Customer</strong> to get a prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model stats row ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("Best Model",   metadata['model_name'].replace(" ", "\u00a0"), ""),
    ("Accuracy",     f"{metadata['accuracy']*100:.1f}%", "on test set"),
    ("ROC-AUC",      f"{metadata['roc_auc']:.3f}", "discrimination"),
    ("F1 Score",     f"{metadata['f1_score']:.3f}", "churn class"),
]
for col, (label, val, sub) in zip([c1,c2,c3,c4], stats):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.4rem">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "About"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if not predict_btn:
        st.markdown("""
        <div class="info-box">
            ← Configure the customer profile in the sidebar and click <strong>Analyse Customer</strong> to generate a churn prediction with SHAP explanation.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Process input
        X_input    = engineer_features(raw_input)
        X_scaled   = pd.DataFrame(scaler.transform(X_input), columns=feature_names)
        churn_prob = model.predict_proba(X_scaled)[0][1]
        prediction = int(churn_prob >= threshold)

        # Risk level
        if churn_prob < 0.30:
            risk_class, risk_text, risk_color = "risk-low",      "LOW RISK",      "#4ade80"
        elif churn_prob < 0.55:
            risk_class, risk_text, risk_color = "risk-medium",   "MEDIUM RISK",   "#facc15"
        elif churn_prob < 0.75:
            risk_class, risk_text, risk_color = "risk-high",     "HIGH RISK",     "#f97316"
        else:
            risk_class, risk_text, risk_color = "risk-critical", "CRITICAL RISK", "#ef4444"

        col_gauge, col_details = st.columns([1, 2], gap="large")

        # ── Gauge ──────────────────────────────────────────────────────────────
        with col_gauge:
            st.markdown(f"""
            <div class="risk-container">
                <div class="risk-label">Churn Probability</div>
                <div class="risk-score" style="color:{risk_color}">{churn_prob*100:.1f}%</div>
                <div class="risk-verdict {risk_class}">{risk_text}</div>
                <div style="margin-top:1.5rem;font-size:0.78rem;color:#6b7280;">
                    Decision threshold: <span style="color:#c4874a">{threshold:.2f}</span>
                </div>
                <div style="margin-top:0.3rem;font-size:0.78rem;color:#6b7280;">
                    Prediction: <span style="color:{'#ef4444' if prediction else '#4ade80'};font-weight:600;">
                    {'⚠ Will Churn' if prediction else '✓ Will Stay'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar, ax_bar = plt.subplots(figsize=(3, 0.6))
            fig_bar.patch.set_facecolor('#161820')
            ax_bar.set_facecolor('#161820')
            ax_bar.barh(0, 1, color='#2a2d3a', height=0.5)
            ax_bar.barh(0, churn_prob, color=risk_color, height=0.5)
            ax_bar.axvline(threshold, color='#c4874a', linewidth=1.5, linestyle='--')
            ax_bar.set_xlim(0, 1)
            ax_bar.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()
            st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.62rem;color:#6b7280;text-align:center;'>0% ── threshold: {threshold:.0%} ── 100%</div>",
                        unsafe_allow_html=True)

        # ── Key risk factors ────────────────────────────────────────────────────
        with col_details:
            st.markdown('<div class="section-title">Key Risk Factors (SHAP)</div>',
                        unsafe_allow_html=True)

            try:
                # Determine explainer type
                model_name = metadata['model_name']
                if model_name in ['XGBoost', 'LightGBM', 'Random Forest',
                                   'Soft Voting', 'Weighted Soft Voting']:
                    # Use LightGBM from all_models for SHAP
                    shap_m = all_models.get('LightGBM', model)
                    explainer   = shap.TreeExplainer(shap_m)
                    shap_vals   = explainer.shap_values(X_scaled)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                else:
                    explainer   = shap.TreeExplainer(all_models['XGBoost'])
                    shap_vals   = explainer.shap_values(X_scaled)

                sv = shap_vals[0]

                # Top 10 features
                top_idx  = np.argsort(np.abs(sv))[-10:][::-1]
                top_feat = [feature_names[i] for i in top_idx]
                top_vals = [sv[i] for i in top_idx]

                fig_shap, ax = plt.subplots(figsize=(6, 4))
                fig_shap.patch.set_facecolor('#161820')
                ax.set_facecolor('#161820')

                colors_shap = ['#ef4444' if v > 0 else '#4ade80' for v in top_vals]
                bars = ax.barh(range(len(top_feat)), top_vals,
                               color=colors_shap, alpha=0.85, edgecolor='none')
                ax.set_yticks(range(len(top_feat)))
                ax.set_yticklabels([f.replace('_', ' ') for f in top_feat],
                                   color='#e8e4dc', fontsize=9)
                ax.set_xlabel('SHAP Value (impact on churn probability)',
                              color='#6b7280', fontsize=8)
                ax.tick_params(colors='#6b7280', labelsize=8)
                ax.axvline(0, color='#2a2d3a', linewidth=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for spine in ['left','bottom']:
                    ax.spines[spine].set_color('#2a2d3a')

                red_patch   = mpatches.Patch(color='#ef4444', label='Increases churn risk')
                green_patch = mpatches.Patch(color='#4ade80', label='Decreases churn risk')
                ax.legend(handles=[red_patch, green_patch], loc='lower right',
                          facecolor='#161820', edgecolor='#2a2d3a',
                          labelcolor='#e8e4dc', fontsize=8)

                plt.tight_layout()
                st.pyplot(fig_shap, use_container_width=True)
                plt.close()

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # ── Customer summary ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Customer Profile Summary</div>',
                    unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        summary_items = [
            ("Tenure",          f"{tenure} months"),
            ("Monthly Charges", f"${monthly_charges}"),
            ("Contract",        contract),
            ("Internet",        internet_service),
            ("Services",        f"{raw_input.get('NumServices', 'N/A')} active"),
            ("Tech Support",    tech_support),
            ("Online Security", online_security),
            ("Payment",         payment_method.split('(')[0].strip()),
        ]
        cols = [s1, s2, s3, s4]
        for i, (label, val) in enumerate(summary_items):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:#161820;border:1px solid #2a2d3a;border-radius:8px;
                            padding:0.8rem 1rem;margin-bottom:0.6rem;">
                    <div style="font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.15em;
                                text-transform:uppercase;color:#6b7280;">{label}</div>
                    <div style="font-size:0.88rem;color:#e8e4dc;font-weight:500;margin-top:0.2rem;">{val}</div>
                </div>""", unsafe_allow_html=True)

        # ── Recommendation ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            rec_color = "#ef4444"
            if churn_prob >= 0.75:
                rec = "🚨 <strong>Immediate action required.</strong> Offer a contract upgrade with a significant discount. Assign a dedicated account manager. Consider a personalised retention call within 24 hours."
            else:
                rec = "⚠️ <strong>Proactive retention recommended.</strong> Consider offering a loyalty discount or upgrading their service tier. Follow up within the week."
        else:
            rec_color = "#4ade80"
            rec = "✅ <strong>Customer appears stable.</strong> Continue standard engagement. Consider upselling additional services given their low churn risk."

        st.markdown(f"""
        <div style="background:rgba(196,135,74,0.06);border:1px solid rgba(196,135,74,0.2);
                    border-left:4px solid {rec_color};border-radius:8px;padding:1.2rem 1.4rem;">
            <div style="font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.15em;
                        text-transform:uppercase;color:#6b7280;margin-bottom:0.5rem;">Recommendation</div>
            <div style="font-size:0.88rem;color:#e8e4dc;line-height:1.7;">{rec}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">All Model Results</div>',
                unsafe_allow_html=True)

    all_results = metadata.get('all_results', {})
    if all_results:
        best_auc = max(v['roc_auc'] for v in all_results.values())

        rows = ""
        for name, res in all_results.items():
            is_winner = res['roc_auc'] == best_auc
            winner_class = 'winner' if is_winner else ''
            badge = '<span class="badge">Best</span>' if is_winner else ''
            rows += f"""
            <tr class="{winner_class}">
                <td>{name}{badge}</td>
                <td>{res['accuracy']*100:.1f}%</td>
                <td>{res['roc_auc']:.4f}</td>
                <td>{res['f1']:.4f}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#161820;border:1px solid #2a2d3a;border-radius:12px;
                    padding:1.5rem;overflow-x:auto;">
            <table class="model-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>ROC-AUC</th>
                        <th>F1-Churn</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Visual Comparison</div>',
                    unsafe_allow_html=True)

        names_list = list(all_results.keys())
        accs  = [all_results[n]['accuracy'] for n in names_list]
        aucs  = [all_results[n]['roc_auc']  for n in names_list]
        f1s   = [all_results[n]['f1']        for n in names_list]

        fig_comp, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig_comp.patch.set_facecolor('#0f1117')

        for ax, vals, title, color in zip(
            axes,
            [accs, aucs, f1s],
            ['Accuracy', 'ROC-AUC', 'F1 Score (Churn)'],
            ['#2e6b7a', '#c4874a', '#4a7c59']
        ):
            ax.set_facecolor('#161820')
            bar_colors = [color if v != max(vals) else '#e8e4dc' for v in vals]
            bars = ax.barh(names_list, vals, color=bar_colors, alpha=0.85, edgecolor='none')
            ax.set_title(title, color='#e8e4dc', fontsize=10, fontweight='bold', pad=10)
            ax.tick_params(colors='#6b7280', labelsize=8)
            ax.set_xlim(min(vals)*0.95, max(vals)*1.03)
            for spine in ax.spines.values():
                spine.set_color('#2a2d3a')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', ha='left', color='#6b7280', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_comp, use_container_width=True)
        plt.close()

    else:
        st.info("Model comparison data not found in metadata.pkl")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-title">About This Project</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.88rem;color:#9ca3af;line-height:1.85;">
        <p>This application predicts customer churn for a telecommunications company using
        an end-to-end machine learning pipeline trained on the
        <strong style="color:#e8e4dc">IBM Telco Customer Churn dataset</strong>.</p>

        <p>Six models were trained and evaluated: Logistic Regression, Random Forest,
        XGBoost, LightGBM, K-Nearest Neighbours, and a Neural Network (MLP). The best
        performing model was selected by cross-validated ROC-AUC score.</p>

        <p>The decision threshold was optimised post-training to maximise accuracy on
        the held-out test set rather than using the default 0.5.</p>

        <p>SHAP (SHapley Additive exPlanations) values provide per-prediction
        explanations showing which features pushed the model toward or away from
        predicting churn.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">Pipeline Overview</div>',
                    unsafe_allow_html=True)
        steps = [
            ("01", "Data Loading & EDA",          "Telco dataset · 7,043 customers · 21 features"),
            ("02", "Feature Engineering",          "12 new features including spend trends & service counts"),
            ("03", "SMOTE Balancing",               "Synthetic oversampling of minority churn class (~26%)"),
            ("04", "Hyperparameter Tuning",         "RandomizedSearchCV · 5-fold stratified CV per model"),
            ("05", "Hybrid Voting Ensemble",        "Soft & weighted soft voting on top tree models"),
            ("06", "Threshold Optimisation",        "Best decision threshold found on held-out test set"),
            ("07", "SHAP Explainability",           "Per-prediction feature attribution for transparency"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start;">
                <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#c4874a;
                            background:rgba(196,135,74,0.1);border:1px solid rgba(196,135,74,0.3);
                            border-radius:4px;padding:0.2rem 0.5rem;white-space:nowrap;
                            margin-top:0.1rem;">{num}</div>
                <div>
                    <div style="font-size:0.88rem;color:#e8e4dc;font-weight:500;">{title}</div>
                    <div style="font-size:0.78rem;color:#6b7280;margin-top:0.1rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
    techs = ["Python 3.11", "Streamlit", "Scikit-learn", "XGBoost", "LightGBM",
             "SHAP", "SMOTE (imbalanced-learn)", "Pandas", "NumPy", "Matplotlib"]
    badges = "".join([
        f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;background:#161820;'
        f'border:1px solid #2a2d3a;border-radius:6px;padding:0.3rem 0.8rem;'
        f'color:#e8e4dc;margin:0.25rem;display:inline-block;">{t}</span>'
        for t in techs
    ])
    st.markdown(f'<div style="line-height:2.5;">{badges}</div>', unsafe_allow_html=True)
