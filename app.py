import streamlit as st
import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt
st.set_page_config(page_title="Heart — Warehouse Dashboard & ML", layout="wide")
st.title("Heart Disease Warehouse — Dashboard, ML & Explainability")

conn = sqlite3.connect("warehouse.db", check_same_thread=False)
@st.cache_data
def load_tables():
    df_fact = pd.read_sql_query("SELECT * FROM fact_exam", conn)
    df_pat = pd.read_sql_query("SELECT * FROM dim_patient", conn)
    df_test = pd.read_sql_query("SELECT * FROM dim_test", conn)
    return df_fact, df_pat, df_test
df_fact, df_pat, df_test = load_tables()
df = df_fact.merge(df_pat, on="patient_sk").merge(df_test, on="test_sk")

tabs = st.tabs(["Overview","Exploration","Cohorts","ML & Explainability"])
with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fact Rows", len(df_fact))
    c2.metric("Unique Patients", df_pat["patient_sk"].nunique())
    c3.metric("Unique Test Profiles", df_test["test_sk"].nunique())
    c4.metric("Heart Disease Rate (%)", f"{df['target'].mean()*100:.1f}%")
    st.markdown("---")
    st.subheader("Data Model")
    st.write("- staging_heart, staging_heart_norm (staging)\n- dim_patient, dim_test (dimensions)\n- fact_exam (fact table)")
with tabs[1]:
    st.header("Exploration")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader('Age distribution')
        fig1 = plt.figure()
        df['age'].plot.hist(bins=15)
        plt.xlabel('Age')
        st.pyplot(fig1)
    with col2:
        st.subheader('Target by Sex')
        pivot = df.groupby('sex')['target'].mean().reset_index()
        fig2 = plt.figure()
        plt.bar(pivot['sex'], pivot['target'])
        plt.ylabel('Proportion target=1')
        st.pyplot(fig2)
    st.markdown('---')
    st.subheader('Data sample')
    st.dataframe(df.head(200))
with tabs[2]:
    st.header("Cohort Analysis")
    bins=[20,30,40,50,60,70,80,100]; labels=["20s","30s","40s","50s","60s","70s","80+"]
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    st.subheader("Heart Disease Rate by Age Cohort")
    age_grouped = df.groupby("age_group")["target"].mean().reset_index()
    fig1,ax1 = plt.subplots(); ax1.bar(age_grouped["age_group"].astype(str), age_grouped["target"]); ax1.set_ylabel("Heart Disease Rate"); st.pyplot(fig1)
    st.subheader("Heart Disease Rate by Sex")
    sex_grouped = df.groupby("sex")["target"].mean().reset_index()
    fig2,ax2 = plt.subplots(); ax2.bar(sex_grouped["sex"], sex_grouped["target"]); ax2.set_ylabel("Heart Disease Rate"); st.pyplot(fig2)
    st.subheader("Heart Disease Rate by Chest Pain Type (cp)")
    cp_grouped = df.groupby("cp")["target"].mean().reset_index()
    fig3,ax3 = plt.subplots(); ax3.bar(cp_grouped["cp"], cp_grouped["target"]); ax3.set_ylabel("Heart Disease Rate"); plt.xticks(rotation=45); st.pyplot(fig3)
with tabs[3]:
    st.header("ML Model, Predictions & Explainability")
    st.sidebar.header("ML Filters (affect training data)")
    age_range = st.sidebar.slider("Age range", int(df.age.min()), int(df.age.max()), (int(df.age.min()), int(df.age.max())))
    sex_filter = st.sidebar.multiselect("Sex", sorted(df.sex.unique()), default=sorted(df.sex.unique()))
    cp_filter = st.sidebar.multiselect("Chest Pain Type (cp)", sorted(df.cp.unique()), default=sorted(df.cp.unique()))
    target_filter = st.sidebar.multiselect("Target", sorted(df.target.unique()), default=sorted(df.target.unique()))
    df_filtered = df[(df.age.between(age_range[0], age_range[1])) & (df.sex.isin(sex_filter)) & (df.cp.isin(cp_filter)) & (df.target.isin(target_filter))]
    st.subheader("Filtered dataset for ML (preview)"); st.write(f"Rows: {len(df_filtered)}"); st.dataframe(df_filtered.head(100))
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    try:
        from xgboost import XGBClassifier; xgb_available = True
    except Exception:
        xgb_available = False
    ml_df = df_filtered.copy()
    if 'age_group' in ml_df.columns:
        ml_df = ml_df.drop('age_group', axis=1)
    ml_df['sex'] = ml_df['sex'].map({'male':1,'female':0})
    ml_df = pd.get_dummies(ml_df, columns=['cp','restecg','slope','thal'], drop_first=True)
    drop_cols = ['patient_sk','test_sk','exam_sk']; ml_df = ml_df.drop([c for c in drop_cols if c in ml_df.columns], axis=1)
    if 'target' not in ml_df.columns:
        st.error("Target column missing after filters. Cannot train.")
    else:
        X = ml_df.drop('target', axis=1); y = ml_df['target']
        if len(X) < 20:
            st.warning("Not enough rows to train reliably. Need at least 20 rows.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            scaler = StandardScaler(); num_cols = [c for c in ['age','trestbps','chol','thalach','oldpeak'] if c in X.columns]
            X_train_scaled = X_train.copy(); X_test_scaled = X_test.copy()
            if num_cols:
                X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]); X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
            model_choice = st.selectbox("Choose model", ["Logistic Regression"] + (["XGBoost"] if xgb_available else []))
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=300); model.fit(X_train_scaled, y_train)
            else:
                model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=200, use_label_encoder=False, eval_metric='logloss'); model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled); y_proba = model.predict_proba(X_test_scaled)[:,1]
            acc = accuracy_score(y_test, y_pred); auc = roc_auc_score(y_test, y_proba)
            st.metric("Test Accuracy", f"{acc:.3f}"); st.metric("Test ROC-AUC", f"{auc:.3f}")
