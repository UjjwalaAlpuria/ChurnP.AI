# =============================================================================
# CUSTOMER CHURN PREDICTION — Streamlit App
# Converted from Spyder script (churn_spyder.py)
# Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn ML Pipeline",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@300;400;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0e17;
    color: #e0e6f0;
}
h1 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 3px; font-size: 3rem !important; color: #e0e6f0; }
h2 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; font-size: 2rem !important; color: #c8d8f0; }
h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1.5px; color: #a8c0e8; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #080c14; border-right: 1px solid #1e2d4a; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1929 0%, #0f2040 100%);
    border: 1px solid #1e3560; border-radius: 10px; padding: 1rem;
}
[data-testid="metric-container"] label { color: #6888aa; font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #58a8ff; font-family: 'DM Mono', monospace; font-size: 2rem; }
.stButton > button {
    background: linear-gradient(90deg, #0055cc, #0088ff);
    color: white; font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem; letter-spacing: 2px;
    border: none; border-radius: 6px; padding: 0.55rem 2rem;
}
.stButton > button:hover { background: linear-gradient(90deg, #0066dd, #00aaff); }
.chip {
    display: inline-block; background: #0d2040; border: 1px solid #1e4080;
    color: #58a8ff; border-radius: 4px; padding: 2px 10px;
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
}
hr { border-color: #1a2840; }
</style>
""", unsafe_allow_html=True)

# ── Plot defaults ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1929", "axes.facecolor": "#0d1929",
    "axes.edgecolor": "#1e3560", "axes.labelcolor": "#7899bb",
    "xtick.color": "#5577aa", "ytick.color": "#5577aa",
    "text.color": "#c0d4ee", "grid.color": "#1a2840",
    "grid.linestyle": "--", "grid.alpha": 0.5, "axes.grid": True,
    "font.family": "monospace",
})
ACCENT  = "#58a8ff"
ACCENT2 = "#00e5a0"
WARN    = "#ff6b6b"
GOLD    = "#ffcc44"

# ── Session helpers ───────────────────────────────────────────────────────────
def sget(key, default=None):
    return st.session_state.get(key, default)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 CHURN PIPELINE")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home & Upload",
        "📋  Data Overview",
        "📊  EDA",
        "🔧  Preprocessing",
        "🤖  Model Training",
        "🏆  Leaderboard",
    ], label_visibility="collapsed")
    st.markdown("---")
    df_tmp = sget("df")
    if df_tmp is not None:
        st.markdown(f"**Dataset** `{df_tmp.shape[0]:,}` × `{df_tmp.shape[1]}`")
    splits_tmp = sget("splits")
    if splits_tmp is not None:
        Xtr, Xte, _, _ = splits_tmp
        st.markdown(f"**Split** `{Xtr.shape[0]}` / `{Xte.shape[0]}`")
    results_tmp = sget("model_results")
    if results_tmp:
        best_s = max(results_tmp, key=results_tmp.get)
        st.markdown(f"**Best** `{best_s[:12]}…` `{results_tmp[best_s]:.4f}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<div class="chip">Customer Churn · ML Pipeline</div>', unsafe_allow_html=True)
    st.title("Churn Prediction\nDashboard")
    st.markdown("A complete end-to-end ML pipeline — from raw Excel data to a trained model leaderboard. Upload your dataset and navigate via the sidebar.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset rows","5,000")
    c2.metric("Features","21")
    c3.metric("Target","churn")
    c4.metric("Models","6")

    st.markdown("---")
    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader("Upload Excel or CSV (e.g. P670-dataset.xlsx)", type=["xlsx","csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state["df"] = df
            for k in ["df_processed","splits","target_col","model_results"]:
                st.session_state.pop(k, None)
            st.success(f"✅  Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
        except Exception as e:
            st.error(f"Failed to load: {e}")
    elif sget("df") is not None:
        st.info("Dataset already loaded — navigate using the sidebar.")

    st.markdown("---")
    st.markdown("### Pipeline Stages")
    for icon, name, desc in [
        ("📋","Data Overview","Shape, dtypes, nulls, descriptive stats"),
        ("📊","EDA","Distributions, boxplots, heatmap, pairplot"),
        ("🔧","Preprocessing","Outlier removal → capping → encoding → scaling → split"),
        ("🤖","Model Training","Train & evaluate 6 classifiers interactively"),
        ("🏆","Leaderboard","Side-by-side accuracy comparison"),
    ]:
        st.markdown(f"**{icon} {name}** — {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif "Overview" in page:
    st.markdown('<div class="chip">Step 1 · Inspection</div>', unsafe_allow_html=True)
    st.title("Data Overview")
    df = sget("df")
    if df is None:
        st.warning("⬅️  Upload your dataset on the Home page first."); st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Null cells", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.markdown("#### First 5 rows")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("#### Column types & null counts")
    st.dataframe(pd.DataFrame({
        "Column": df.columns, "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values, "Null": df.isnull().sum().values,
    }), use_container_width=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Missing value heatmap")
    fig, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(df.isnull(), cbar=False, ax=ax, cmap="YlOrRd", yticklabels=False)
    ax.set_title("Yellow = missing values", fontsize=11)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="chip">Step 2 · Exploratory Analysis</div>', unsafe_allow_html=True)
    st.title("Exploratory Data Analysis")
    df = sget("df")
    if df is None:
        st.warning("⬅️  Upload your dataset on the Home page first."); st.stop()

    numerical_col = df.select_dtypes(include=np.number).columns.tolist()

    # Churn distribution
    if "churn" in df.columns:
        st.markdown("#### Churn Distribution")
        counts = df["churn"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(counts.index, counts.values, color=[ACCENT, WARN], edgecolor="none", width=0.5)
        axes[0].set_title("Churn Counts")
        axes[1].pie(counts.values, labels=counts.index, colors=[ACCENT, WARN],
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor":"#0d1929","linewidth":2})
        axes[1].set_title("Churn Split")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # KDE histogram
    st.markdown("#### Feature Distribution (KDE)")
    col_sel = st.selectbox("Select column", numerical_col)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[col_sel].dropna(), kde=True, ax=ax,
                 color=ACCENT, line_kws={"color": ACCENT2, "lw": 2})
    ax.set_title(f"Distribution — {col_sel}", fontweight="bold")
    st.pyplot(fig); plt.close()

    # All histograms
    with st.expander("📊 Show all feature histograms"):
        fig = df[numerical_col].hist(figsize=(18,14), bins=25,
                                     color=ACCENT, edgecolor="#0d1929", grid=False)
        plt.suptitle("All Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.close()

    # Boxplots
    st.markdown("#### Boxplots — Outlier Overview")
    fig, ax = plt.subplots(figsize=(18, 6))
    df[numerical_col].boxplot(ax=ax, patch_artist=True,
        boxprops=dict(facecolor="#0d2040", color=ACCENT),
        medianprops=dict(color=ACCENT2, lw=2),
        whiskerprops=dict(color=ACCENT), capprops=dict(color=ACCENT),
        flierprops=dict(marker="o", color=WARN, alpha=0.4, markersize=3))
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Boxplots (raw data)", fontweight="bold")
    st.pyplot(fig); plt.close()

    # Summary stats
    st.markdown("#### Summary Statistics Table")
    summ = pd.DataFrame()
    for col in numerical_col:
        summ.loc[col,"Mean"]   = df[col].mean()
        summ.loc[col,"Median"] = df[col].median()
        summ.loc[col,"Mode"]   = df[col].mode()[0]
        summ.loc[col,"Std"]    = df[col].std()
        q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75)
        summ.loc[col,"IQR"]    = q3 - q1
    st.dataframe(summ.round(2), use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df[numerical_col].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                ax=ax, linewidths=0.3, annot_kws={"size":7}, linecolor="#0a0e17")
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    st.pyplot(fig); plt.close()

    # Pairplot
    st.markdown("#### Pairplot (300-row sample)")
    pair_cols = st.multiselect("Choose columns (2–5 recommended)",
                               numerical_col, default=numerical_col[:4])
    if len(pair_cols) >= 2:
        sample = df[pair_cols].dropna().sample(min(300, len(df)), random_state=42)
        with st.spinner("Rendering pairplot..."):
            fig = sns.pairplot(sample, height=2.2,
                               plot_kws={"color":ACCENT,"alpha":0.5,"s":10},
                               diag_kws={"color":ACCENT2})
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif "Preprocessing" in page:
    st.markdown('<div class="chip">Step 3 · Feature Engineering</div>', unsafe_allow_html=True)
    st.title("Preprocessing Pipeline")
    df = sget("df")
    if df is None:
        st.warning("⬅️  Upload your dataset on the Home page first."); st.stop()

    st.markdown("""
    Runs all steps from the original notebook in sequence:
    1. **Outlier removal** via IQR filtering
    2. **Outlier capping** (Winsorisation)
    3. **Feature engineering** — aggregate minutes, calls & charges
    4. **One-hot encoding** of all categorical columns
    5. **Min-Max scaling** to [0, 1]
    6. **Train / test split** (80 / 20, stratified)
    """)

    if st.button("▶  RUN FULL PIPELINE"):
        with st.spinner("Running pipeline..."):
            df_proc       = df.copy()
            numerical_col = df_proc.select_dtypes(include=np.number).columns
            initial_shape = df_proc.shape

            # 1. Outlier removal
            for col in numerical_col:
                Q1=df_proc[col].quantile(0.25); Q3=df_proc[col].quantile(0.75); IQR=Q3-Q1
                df_proc = df_proc[(df_proc[col] >= Q1-1.5*IQR) & (df_proc[col] <= Q3+1.5*IQR)]
            after_removal = df_proc.shape

            # 2. Outlier capping
            for col in df_proc.select_dtypes(include=np.number).columns:
                Q1=df_proc[col].quantile(0.25); Q3=df_proc[col].quantile(0.75); IQR=Q3-Q1
                lb=Q1-1.5*IQR; ub=Q3+1.5*IQR
                df_proc[col]=np.where(df_proc[col]<lb,lb,df_proc[col])
                df_proc[col]=np.where(df_proc[col]>ub,ub,df_proc[col])

            # 3. Feature engineering
            fe_done = []
            try:
                df_proc["total_call_minutes"] = df_proc["day.mins"]+df_proc["night.mins"]+df_proc["intl.mins"]
                df_proc["total_calls"]        = df_proc["day.calls"]+df_proc["night.calls"]+df_proc["intl.calls"]+df_proc["eve.calls"]
                df_proc["total_charge"]       = df_proc["night.charge"]+df_proc["intl.charge"]+df_proc["eve.charge"]
                fe_done = ["total_call_minutes","total_calls","total_charge"]
            except KeyError:
                pass

            # 4. One-hot encoding
            cat_cols = df_proc.select_dtypes(include="object").columns.tolist()
            df_proc  = pd.get_dummies(df_proc, columns=cat_cols)

            # 5. Min-Max scaling
            mn        = MinMaxScaler()
            df_scaled = pd.DataFrame(mn.fit_transform(df_proc), columns=df_proc.columns)

            # 6. Train/test split
            target_col = next((c for c in df_scaled.columns if "churn_yes" in c), None)
            if target_col is None:
                st.error("Could not find 'churn_yes' after encoding. Check your churn column name."); st.stop()

            churn_cols = [c for c in df_scaled.columns if "churn" in c]
            X = df_scaled.drop(columns=churn_cols)
            y = df_scaled[target_col]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

            st.session_state.update({
                "df_processed": df_scaled,
                "splits":       (X_train,X_test,y_train,y_test),
                "target_col":   target_col,
            })
            if "model_results" not in st.session_state:
                st.session_state["model_results"] = {}

        st.success("✅  Pipeline complete!")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Original rows",         f"{initial_shape[0]:,}")
        c2.metric("After outlier removal", f"{after_removal[0]:,}")
        c3.metric("Rows removed",          f"{initial_shape[0]-after_removal[0]:,}")
        c4.metric("Features",              X_train.shape[1])

        if fe_done:
            st.info(f"🔬 Engineered: `{'`, `'.join(fe_done)}`")

        # Boxplot after capping
        st.markdown("#### Boxplots After Capping")
        num_proc = df_proc.select_dtypes(include=np.number)
        fig, ax  = plt.subplots(figsize=(18,6))
        num_proc.boxplot(ax=ax, patch_artist=True,
            boxprops=dict(facecolor="#0d2040",color=ACCENT2),
            medianprops=dict(color=ACCENT,lw=2),
            whiskerprops=dict(color=ACCENT2), capprops=dict(color=ACCENT2),
            flierprops=dict(marker="o",color=WARN,alpha=0.3,markersize=2))
        plt.xticks(rotation=45, ha="right", fontsize=7)
        ax.set_title("Boxplots after outlier removal + capping", fontweight="bold")
        st.pyplot(fig); plt.close()

        # Correlation with target bar chart
        st.markdown("#### Correlation with `churn_yes`")
        corrs = df_scaled.corrwith(df_scaled[target_col]).drop(target_col).sort_values(ascending=False)
        top   = pd.concat([corrs.head(10), corrs.tail(10)])
        colors_bar = [ACCENT2 if v >= 0 else WARN for v in top.values]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.barh(top.index, top.values, color=colors_bar, edgecolor="none", height=0.6)
        ax.axvline(0, color="#4a6080", lw=1)
        ax.set_title("Top 10 +ve / −ve correlations with churn", fontweight="bold")
        ax.invert_yaxis(); plt.tight_layout(); st.pyplot(fig); plt.close()

        c1,c2 = st.columns(2)
        c1.metric("Training samples", X_train.shape[0])
        c2.metric("Test samples",     X_test.shape[0])
        st.markdown("#### Scaled DataFrame (first 5 rows)")
        st.dataframe(df_scaled.head(), use_container_width=True)

    elif sget("splits") is not None:
        st.info("✅  Preprocessing already complete. Proceed to Model Training.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="chip">Step 4 · Model Training</div>', unsafe_allow_html=True)
    st.title("Model Training & Evaluation")

    splits = sget("splits")
    if splits is None:
        st.warning("⬅️  Run Preprocessing first."); st.stop()

    X_train, X_test, y_train, y_test = splits

    model_choice = st.selectbox("Select a model", [
        "Logistic Regression","SVM (RBF kernel)","Decision Tree",
        "Random Forest","K-Nearest Neighbors","XGBoost",
    ])

    with st.expander("⚙️  Hyperparameters"):
        if model_choice == "Logistic Regression":
            max_iter = st.slider("max_iter", 100, 2000, 1000, 100)
            C_lr     = st.select_slider("C", [0.001,0.01,0.1,1,10,100], value=1)
        elif model_choice == "SVM (RBF kernel)":
            C_svm  = st.select_slider("C", [0.1,1,5,10,50,100], value=10)
            gamma  = st.select_slider("gamma", [0.001,0.01,0.1,1,10], value=0.001)
        elif model_choice == "Decision Tree":
            max_depth = st.slider("max_depth", 1, 10, 2)
            criterion = st.radio("criterion", ["entropy","gini"], horizontal=True)
        elif model_choice == "Random Forest":
            n_est    = st.slider("n_estimators", 50, 500, 100, 50)
            max_d_rf = st.slider("max_depth", 2, 20, 10)
        elif model_choice == "K-Nearest Neighbors":
            k_val = st.slider("n_neighbors (K)", 1, 21, 5, 2)
        elif model_choice == "XGBoost":
            n_est_xgb = st.slider("n_estimators", 50, 500, 100, 50)
            lr_xgb    = st.select_slider("learning_rate", [0.01,0.05,0.1,0.2,0.3], value=0.1)
            md_xgb    = st.slider("max_depth", 2, 10, 6)

    if st.button(f"▶  TRAIN {model_choice.upper()}"):
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=max_iter, C=C_lr)
            elif model_choice == "SVM (RBF kernel)":
                model = SVC(kernel="rbf", C=C_svm, gamma=gamma)
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d_rf, random_state=42)
            elif model_choice == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=k_val)
            elif model_choice == "XGBoost":
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                         use_label_encoder=False, n_estimators=n_est_xgb,
                                         learning_rate=lr_xgb, max_depth=md_xgb, random_state=42)
                except ImportError:
                    st.error("XGBoost not installed. Run: `pip install xgboost`"); st.stop()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm     = confusion_matrix(y_test, y_pred)

            if "model_results" not in st.session_state:
                st.session_state["model_results"] = {}
            st.session_state["model_results"][model_choice] = acc

        st.success(f"✅  {model_choice} trained!")
        st.markdown(f"### Test Accuracy — `{acc:.4f}`")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{model_choice}", fontweight="bold")
            st.pyplot(fig); plt.close()

        with c2:
            st.markdown("#### Classification Report")
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        # Decision Tree plot
        if model_choice == "Decision Tree":
            st.markdown("#### Decision Tree Visualisation")
            fig, ax = plt.subplots(figsize=(16, 6))
            plot_tree(model, feature_names=X_train.columns,
                      class_names=["No Churn","Churn"],
                      filled=True, ax=ax, fontsize=8)
            st.pyplot(fig); plt.close()

        # Feature importances
        if model_choice in ["Random Forest","Decision Tree","XGBoost"]:
            st.markdown("#### Top 20 Feature Importances")
            fi = pd.Series(model.feature_importances_, index=X_train.columns).nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(9, 7))
            bars = ax.barh(fi.index, fi.values, color=ACCENT2, edgecolor="none", height=0.65)
            for bar, val in zip(bars, fi.values):
                ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=7)
            ax.set_title("Feature Importances", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # KNN K-curve
        if model_choice == "K-Nearest Neighbors":
            st.markdown("#### KNN — Accuracy vs K")
            with st.spinner("Computing K-curve..."):
                K_range  = range(1, 20, 2)
                K_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_train, y_train, cv=KFold(n_splits=5),
                             scoring="accuracy").mean() for k in K_range]
            best_k = list(K_range)[int(np.argmax(K_scores))]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(list(K_range), K_scores, marker="o", color=ACCENT, lw=2, markersize=6)
            ax.axvline(best_k, color=ACCENT2, ls="--", lw=1.5, label=f"Best K={best_k}")
            ax.set_xlabel("K"); ax.set_ylabel("CV Accuracy")
            ax.set_title("KNN — Accuracy vs K", fontweight="bold")
            ax.legend(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Leaderboard" in page:
    st.markdown('<div class="chip">Step 5 · Final Comparison</div>', unsafe_allow_html=True)
    st.title("Model Leaderboard")

    baseline = {
        "Logistic Regression" : 0.9192,
        "SVM (RBF kernel)"    : 0.8951,
        "Decision Tree"       : 0.9047,
        "Random Forest"       : 0.9288,
        "K-Nearest Neighbors" : 0.8987,
        "XGBoost"             : 0.9783,
    }
    live     = sget("model_results") or {}
    combined = {**baseline, **live}
    series   = pd.Series(combined).sort_values(ascending=False)
    best     = series.index[0]

    st.info("Baseline values are from the original notebook. Models you train override them.")

    lb = pd.DataFrame({
        "Rank":         range(1, len(series)+1),
        "Model":        series.index,
        "Accuracy":     series.values.round(4),
        "Accuracy (%)": (series.values*100).round(2),
        "Source":       ["🟢 Live" if m in live else "📓 Baseline" for m in series.index],
    })

    def highlight(row):
        if row["Model"] == best:
            return ["background-color:#0d2a00;color:#aadd00;font-weight:bold"]*len(row)
        return [""]*len(row)

    st.dataframe(lb.style.apply(highlight,axis=1),
                 use_container_width=True, hide_index=True, height=280)

    # Bar chart
    st.markdown("#### Accuracy Bar Chart")
    colors_lb = [GOLD if m == best else ACCENT for m in series.index]
    fig, ax   = plt.subplots(figsize=(11, 5))
    bars      = ax.barh(series.index, series.values,
                        color=colors_lb, edgecolor="none", height=0.55)
    ax.set_xlim(0.84, 1.01)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Comparison", fontweight="bold", fontsize=13)
    for bar, val in zip(bars, series.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.invert_yaxis(); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown(f"### 🏆  Winner — **{best}**  `{series[best]*100:.2f}%`")
    st.markdown("""
    **Conclusion:** XGBoost leads all classifiers at ~97.8% accuracy.  
    Random Forest is the best non-boosting option at ~92.9%, offering a good trade-off  
    between performance and interpretability.
    """)
