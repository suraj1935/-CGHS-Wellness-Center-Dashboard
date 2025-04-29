import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CGHS Wellness Dashboard", layout="wide")
st.title("🏥 CGHS Wellness Center Dashboard")

# File upload in sidebar
st.sidebar.header("📁 Upload Data Files")
uploaded_excel = st.sidebar.file_uploader("Upload cleaned dataset (.xlsx)", type=["xlsx"])
uploaded_csv = st.sidebar.file_uploader("Upload beneficiaries data (.csv)", type=["csv"])


@st.cache_data
def load_data(excel_file, csv_file):
    try:
        if excel_file is None or csv_file is None:
            return pd.DataFrame(), pd.DataFrame()

        centers_df = pd.read_excel(excel_file)
        beneficiaries_df = pd.read_csv(csv_file)

        # Remove spaces from column names
        centers_df.columns = centers_df.columns.map(str).str.strip()
        beneficiaries_df.columns = beneficiaries_df.columns.map(str).str.strip()

        return centers_df, beneficiaries_df

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# Load data
centers_df, beneficiaries_df = load_data(uploaded_excel, uploaded_csv)


# Column name cleanup again for safety
if not centers_df.empty:
    centers_df.columns = centers_df.columns.map(str).str.strip()
if not beneficiaries_df.empty:
    beneficiaries_df.columns = beneficiaries_df.columns.map(str).str.strip()


# Validate column presence
def check_required_columns(df, required_columns, df_name):
    if not isinstance(df, pd.DataFrame):
        st.error(f"{df_name} is not a valid DataFrame")
        return False
    if df.empty:
        st.warning(f"⚠️ {df_name} is empty or not uploaded.")
        return False
    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"❌ Missing required columns in {df_name}: {missing}")
        return False
    return True


# Stop execution if files are not uploaded
if uploaded_excel is None or uploaded_csv is None:
    st.info("📂 Please upload both required files to continue.")
    st.stop()

# Validate required columns
if not check_required_columns(centers_df, {'Wellness Center'}, "Centers Data"):
    st.stop()
if not check_required_columns(beneficiaries_df, {'Wellness Center', 'City'}, "Beneficiaries Data"):
    st.stop()

# Success message
st.success("✅ Data loaded successfully!")

# Optional data preview
with st.expander("🔍 Preview Uploaded Data"):
    st.subheader("Centers Data")
    st.dataframe(centers_df.head())
    st.subheader("Beneficiaries Data")
    st.dataframe(beneficiaries_df.head())

# Placeholder for further logic
st.markdown("📊 Add your clustering, charts, or analytics here.")
