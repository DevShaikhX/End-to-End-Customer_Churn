import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import subprocess
import threading

from joblib import load

# Try to import local evaluation utilities (evaluate_models.py)
try:
    import evaluate_models as ev
    EVAL_AVAILABLE = True
except Exception:
    EVAL_AVAILABLE = False

from utils import load_and_clean_base_data


APP_TITLE = "Telco Churn — Model Ops Dashboard"
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
VISUALS_DIR = Path('visuals')
MODELS_DIR = Path('models')
CHAMPION_PATH = MODELS_DIR / 'champion_model.pkl'


def safe_load_champion(path: Path):
    try:
        artifact = load(path)
        return artifact
    except Exception as e:
        st.warning(f'Champion model not found or failed to load: {e}')
        return None


def list_visuals(dirpath: Path):
    try:
        files = sorted(dirpath.glob('*'))
        return [f for f in files if f.is_file()]
    except Exception:
        return []


def inject_style():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        html, body, [data-testid='stAppViewContainer'], [data-testid='stAppViewContainer'] > div:first-child {
            background: radial-gradient(circle at top left, rgba(0, 186, 255, 0.14), transparent 28%),
                        linear-gradient(180deg, #041526 0%, #081d43 100%);
            color: #e9f2ff;
        }
        [data-testid='stSidebar'], [data-testid='stSidebarNav'], .css-1avcm0n, .css-1avcm0n > div {
            background: linear-gradient(180deg, #071a2f 0%, #0b2f56 100%) !important;
            color: #e8f7ff !important;
            border-right: 1px solid rgba(117, 173, 255, 0.14) !important;
            box-shadow: 2px 0 38px rgba(0, 0, 0, 0.16);
        }
        [data-testid='stSidebarNav'] .css-1v0mbdj.e8zbici2, [data-testid='stSidebar'] .css-1v0mbdj.e8zbici2 {
            background-color: rgba(255,255,255,0.08) !important;
        }
        .css-1lcbmhc.e1tzin5v0, .css-1e5imcs.edgvbvh3, .css-144xtw0.e16nr0p31 {
            background: rgba(5, 18, 45, 0.96) !important;
            border: 1px solid rgba(72, 148, 240, 0.2) !important;
            box-shadow: 0 24px 48px rgba(0,0,0,0.22);
        }
        .css-1lcbmhc.e1tzin5v0 h1, .css-1lcbmhc.e1tzin5v0 h2, .css-1lcbmhc.e1tzin5v0 h3,
        .css-1e5imcs.edgvbvh3 h1, .css-1e5imcs.edgvbvh3 h2, .css-1e5imcs.edgvbvh3 h3 {
            color: #f4fbff;
        }
        .big-font {
            font-size: 3rem !important;
            font-weight: 900 !important;
            letter-spacing: -0.04em;
            color: #f2f8ff;
            text-shadow: 0 5px 22px rgba(0, 0, 0, 0.28);
        }
        .section-header {
            color: #b2d4ff;
            font-size: 1.7rem;
            font-weight: 800;
            margin-top: 1rem;
            margin-bottom: 0.6rem;
        }
        .card {
            background: rgba(8, 23, 48, 0.95);
            border: 1px solid rgba(70, 140, 235, 0.18);
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: 0 26px 54px rgba(0, 0, 0, 0.24);
            color: #e7f2ff;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2b9cff, #00d1ff);
            color: #ffffff;
            border-radius: 14px;
            padding: 0.88rem 1.5rem;
            border: none;
            box-shadow: 0 16px 36px rgba(15, 82, 148, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1f85d6, #00b3d4);
            color: #ffffff;
        }
        .stButton>button:focus {
            outline: 2px solid rgba(78, 232, 255, 0.25);
        }
        .stMetricValue {
            color: #ffffff !important;
            font-weight: 700;
        }
        .stMetricLabel {
            color: #9bc8ff !important;
        }
        .streamlit-expanderHeader {
            color: #e3f1ff !important;
        }
        [data-testid='stMarkdownContainer'] p, [data-testid='stMarkdownContainer'] li {
            color: #d9e8ff;
        }
        .stSidebar .css-1avcm0n {
            background-color: transparent;
        }
        .stSidebar .css-1v3fvcr.egzxvld0 {
            background: transparent;
        }
        .css-1n8qwdk.edgvbvh3 {
            color: #d8ecff !important;
        }
        .css-1n76uvr.e19lei0e7 {
            color: #bcdcff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_generate_artifacts_bg(logpath: Path = Path('generate_artifacts_log.txt')):
    """Start generate_artifacts.py in background and return Popen handle."""
    python = sys.executable
    cmd = [python, '-u', 'generate_artifacts.py']
    with open(logpath, 'ab') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    return proc


def wait_for_artifacts(timeout: int = 600, poll_interval: float = 2.0):
    """Wait for expected artifact files to appear, returns True if found."""
    start = time.time()
    expected = [Path('models') / 'champion_model.pkl', Path('visuals') / 'performance_leaderboard.csv']
    while time.time() - start < timeout:
        if all(p.exists() for p in expected):
            return True
        time.sleep(poll_interval)
    return False


def run_training_and_visuals():
    if not EVAL_AVAILABLE:
        st.error('evaluate_models module not available in workspace.')
        return None
    # Non-blocking: spawn background process to run generate_artifacts.py and poll for artifacts
    placeholder = st.empty()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Run Full Training & Generate Visuals'):
            placeholder.info('Starting artifact generation in background...')
            proc = run_generate_artifacts_bg()
            # Poll for artifacts with a spinner
            with st.spinner('Waiting for artifacts (this may take several minutes)...'):
                ok = wait_for_artifacts(timeout=900)
            if ok:
                placeholder.success('Artifacts generated successfully.')
                try:
                    lb = pd.read_csv(Path('visuals') / 'performance_leaderboard.csv', index_col=0)
                    return lb
                except Exception:
                    return None
            else:
                placeholder.error('Timed out waiting for artifacts. Check generate_artifacts_log.txt for details.')
                return None
    with col2:
        st.write('Click the button to start training in background; monitor `generate_artifacts_log.txt`.')


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon='📈', layout='wide')
    inject_style()

    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"<div class='big-font'>{APP_TITLE}</div>", unsafe_allow_html=True)
            st.markdown('#### A polished Telco churn operations dashboard with live model artifacts and fast data insights.')
        with col2:
            st.markdown('### Status')
            st.write('• artifacts generated')
            st.write('• model ready')
            st.write('• visuals available')
    st.markdown('---')

    # Sidebar navigation
    st.sidebar.title('Telco Churn Navigator')
    sections = ['Home Page', 'Dataset Explorer', 'EDA Dashboard', 'Model Training', 'Model Comparison', 'Prediction System Form']
    choice = st.sidebar.selectbox('Navigation', sections)

    # Load dataset lazily
    df = None
    try:
        df = load_and_clean_base_data(DATA_PATH)
    except Exception as e:
        st.sidebar.error(f'Could not load dataset: {e}')

    # Try to load champion artifact
    champion = None
    try:
        if CHAMPION_PATH.exists():
            champion = safe_load_champion(CHAMPION_PATH)
    except Exception:
        champion = None

    if choice == 'Home Page':
        st.markdown("<div class='section-header'>Home</div>", unsafe_allow_html=True)
        st.markdown('''
        A modern MLOps dashboard for Telco churn prediction, designed to help business users review metrics, inspect visuals, and download production-ready artifacts.
        ''')

        st.markdown("<div class='section-header'>Asset Directory</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card"><h4>Models Directory</h4>', unsafe_allow_html=True)
            try:
                models = [p.name for p in MODELS_DIR.glob('*') if p.is_file()]
                if models:
                    for m in models:
                        st.write(f'• {m}')
                else:
                    st.info('No model artifacts present yet.')
            except Exception as e:
                st.error('Error listing models: {}'.format(e))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><h4>Visuals Directory</h4>', unsafe_allow_html=True)
            try:
                visuals = [p.name for p in VISUALS_DIR.glob('*') if p.is_file()]
                if visuals:
                    for v in visuals:
                        st.write(f'• {v}')
                else:
                    st.info('No visuals found. Run Model Training to generate them.')
            except Exception as e:
                st.error('Error listing visuals: {}'.format(e))
            st.markdown('</div>', unsafe_allow_html=True)

        if champion is not None:
            st.markdown("<div class='section-header'>Current Champion</div>", unsafe_allow_html=True)
            try:
                meta = champion.get('metrics', {})
                model_obj = champion.get('model')
                model_name = champion.get('model_name') or (type(model_obj).__name__ if model_obj is not None else 'Unknown')
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric('Champion', model_name)
                    st.metric('F1-Score', f"{meta.get('f1_score', np.nan):.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('**Champion Metrics**')
                    st.table(pd.DataFrame(meta, index=[0]).T.rename(columns={0: 'value'}))
                    st.markdown('</div>', unsafe_allow_html=True)

                links_col1, links_col2 = st.columns(2)
                with links_col1:
                    try:
                        with open(CHAMPION_PATH, 'rb') as f:
                            st.download_button('Download champion_model.pkl', data=f, file_name='champion_model.pkl')
                    except Exception:
                        st.write('Champion artifact not downloadable.')
                with links_col2:
                    lb_path = VISUALS_DIR / 'performance_leaderboard.csv'
                    if lb_path.exists():
                        with open(lb_path, 'rb') as f:
                            st.download_button('Download leaderboard CSV', data=f, file_name='performance_leaderboard.csv')
            except Exception as e:
                st.error(f'Failed to display champion metadata: {e}')

    elif choice == 'Dataset Explorer':
        st.header('Dataset Explorer')
        if df is None:
            st.error('Dataset unavailable. Check logs in sidebar.')
            return

        st.subheader('Raw Data (searchable)')
        st.dataframe(df)

        st.subheader('Statistical Overview')
        try:
            st.write(df.describe(include='all'))
        except Exception as e:
            st.write('Error computing describe():', e)

        st.subheader('Missing / Null Metrics')
        try:
            missing = df.isnull().sum().sort_values(ascending=False)
            missing_pct = (missing / len(df) * 100).round(2)
            miss_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
            st.dataframe(miss_df[miss_df['missing_count'] > 0])
        except Exception as e:
            st.write('Error computing missing metrics:', e)

    elif choice == 'EDA Dashboard':
        st.header('EDA Dashboard')
        st.write('Select a visualization to load from the visuals directory.')
        files = list_visuals(VISUALS_DIR)
        if not files:
            st.info('No visuals found. Run Model Training to generate charts.')
        else:
            sel = st.selectbox('Choose visual', [f.name for f in files])
            file_path = VISUALS_DIR / sel
            try:
                if sel.lower().endswith(('.png', '.jpg', '.jpeg')):
                    st.image(str(file_path), use_column_width=True)
                elif sel.lower().endswith('.html'):
                    html = file_path.read_text(encoding='utf-8')
                    components.html(html, height=700)
                else:
                    st.write('Preview not supported for this file type. File saved at:', str(file_path))
            except Exception as e:
                st.error(f'Failed to display visual: {e}')

    elif choice == 'Model Training':
        st.markdown("<div class='section-header'>Model Training</div>", unsafe_allow_html=True)
        st.markdown('Use the button below to generate fresh artifacts, metrics, and visuals. This runs in the background and updates the artifacts directory.')
        if not EVAL_AVAILABLE:
            st.error('Evaluation pipeline not available (evaluate_models.py missing).')
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            run_button = st.button('Run Full Training & Generate Visuals')
            st.markdown('</div>', unsafe_allow_html=True)
        if run_button:
            leaderboard = run_training_and_visuals()
            if leaderboard is not None:
                st.success('Training complete — leaderboard updated')
                st.dataframe(leaderboard)

    elif choice == 'Model Comparison':
        st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
        # Load leaderboard CSV if present
        leaderboard_path = VISUALS_DIR / 'performance_leaderboard.csv'
        if leaderboard_path.exists():
            try:
                lb = pd.read_csv(leaderboard_path, index_col=0)
                st.subheader('Performance Leaderboard')
                st.dataframe(lb)
                # Highlight champion if available
                if champion is not None:
                    champ_name = champion.get('model').__class__.__name__
                    st.success(f'Champion: {champ_name}')
            except Exception as e:
                st.write('Failed to load leaderboard:', e)
        else:
            st.info('No leaderboard CSV found. Run Model Training to generate it.')

        # Display common visual assets
        viz_files = list_visuals(VISUALS_DIR)
        grid = [f for f in viz_files if f.name.lower().endswith(('.png', '.jpg'))]
        if grid:
            rows = [grid[i:i + 2] for i in range(0, len(grid), 2)]
            for row in rows:
                cols = st.columns(len(row))
                for col, f in zip(cols, row):
                    with col:
                        st.image(str(f), caption=f.name, use_column_width=True)
        else:
            st.info('No visuals available yet. Run Model Training first.')

    elif choice == 'Prediction System Form':
        st.header('Prediction System')
        st.write('Fill in customer attributes and submit to get a churn prediction from the champion model.')

        if df is None:
            st.error('Dataset required to populate form controls. Please ensure dataset is present.')
            return

        # Build a simple form using dataset feature ranges
        feature_cols = df.drop(columns=['Churn'], errors='ignore')

        with st.form('prediction_form'):
            col1, col2 = st.columns(2)
            with col1:
                tenure = st.slider('tenure', int(feature_cols['tenure'].min()), int(feature_cols['tenure'].max()), int(feature_cols['tenure'].median()))
                monthly = st.slider('MonthlyCharges', int(feature_cols['MonthlyCharges'].min()), int(feature_cols['MonthlyCharges'].max()), int(feature_cols['MonthlyCharges'].median()))
            with col2:
                contract = st.selectbox('Contract', sorted(feature_cols['Contract'].unique().tolist()))
                internet = st.selectbox('InternetService', sorted(feature_cols['InternetService'].unique().tolist()))

            # For remaining categorical fields show a compact multiselect to set defaults
            extra_inputs = {}
            for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'PaymentMethod']:
                if col in feature_cols.columns:
                    opts = sorted(feature_cols[col].dropna().unique().tolist())
                    extra_inputs[col] = st.selectbox(col, opts, index=0)

            submit = st.form_submit_button('Predict')

        if submit:
            if champion is None:
                st.error('Champion model not loaded. Ensure models/champion_model.pkl exists.')
            else:
                try:
                    # Reconstruct a single-row DataFrame matching original feature columns
                    input_row = pd.DataFrame([{**{c: None for c in feature_cols.columns}}])
                    # Fill known fields
                    if 'tenure' in input_row.columns:
                        input_row.at[0, 'tenure'] = tenure
                    if 'MonthlyCharges' in input_row.columns:
                        input_row.at[0, 'MonthlyCharges'] = monthly
                    if 'Contract' in input_row.columns:
                        input_row.at[0, 'Contract'] = contract
                    if 'InternetService' in input_row.columns:
                        input_row.at[0, 'InternetService'] = internet
                    for k, v in extra_inputs.items():
                        input_row.at[0, k] = v

                    # Fill any remaining nulls with mode or median
                    for col in input_row.columns:
                        if pd.isnull(input_row.at[0, col]):
                            if feature_cols[col].dtype.kind in 'biufc':
                                input_row.at[0, col] = int(feature_cols[col].median())
                            else:
                                vals = feature_cols[col].dropna().unique()
                                input_row.at[0, col] = vals[0] if len(vals) > 0 else ''

                    # Transform using stored preprocessor (artifact contains 'preprocessor')
                    pre = champion.get('preprocessor')
                    model = champion.get('model')
                    X_in = pre.transform(input_row)
                    proba = None
                    try:
                        proba = model.predict_proba(X_in)[:, 1][0]
                    except Exception:
                        # Some models don't expose predict_proba
                        pred = model.predict(X_in)[0]
                        proba = float(pred)

                    label = 'High Risk Customer - Attrition Likely' if proba >= 0.5 else 'Stable Safe Profile - Retention High'
                    st.markdown(f"**Prediction:** {label}")
                    st.markdown(f"**Confidence:** {proba * 100:.2f}%")
                except Exception as e:
                    st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    main()
