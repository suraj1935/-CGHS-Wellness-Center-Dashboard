# app.py
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# Streamlit page configuration
st.set_page_config(page_title="CGHS Wellness Dashboard", layout="wide")
st.title("\U0001F3E5 CGHS Wellness Center Dashboard")

# Sidebar for file upload
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

        centers_df.columns = centers_df.columns.str.strip()
        beneficiaries_df.columns = beneficiaries_df.columns.str.strip()

        return centers_df, beneficiaries_df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Load data
centers_df, beneficiaries_df = load_data(uploaded_excel, uploaded_csv)

# Rename to standard column name
if not centers_df.empty:
    for col in centers_df.columns:
        if col.strip() in ['Wellness Center', 'wellnessCentreName', 'WellnessCentreName', 'Center Name']:
            centers_df.rename(columns={col: 'Wellness Center'}, inplace=True)
            break

# Validate required columns
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

# Require file upload to proceed
if uploaded_excel is None or uploaded_csv is None:
    st.info("📂 Please upload both required files to continue.")
    st.stop()

# Column validation
if not check_required_columns(centers_df, {'Wellness Center'}, "Centers Data"):
    st.stop()
if not check_required_columns(beneficiaries_df, {'Wellness Center', 'City'}, "Beneficiaries Data"):
    st.stop()

st.success("✅ Data loaded successfully!")

# Preview data
with st.expander("🔍 Preview Uploaded Data"):
    st.subheader("Centers Data")
    st.dataframe(centers_df.head())
    st.subheader("Beneficiaries Data")
    st.dataframe(beneficiaries_df.head())

# Clustering setup
st.header("📊 KMeans Clustering on Beneficiaries Data")

try:
    grouped_df = beneficiaries_df.groupby('Wellness Center').size().reset_index(name='Beneficiary Count')

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(grouped_df[['Beneficiary Count']])

    # Function to compute WCSS
    def calculate_wcss(data):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        return wcss

    # Elbow method
    st.subheader("📈 Elbow Method (Optional: Determine Optimal Clusters)")
    wcss = calculate_wcss(X_scaled)

    fig_elbow = px.line(
        x=list(range(1, 11)),
        y=wcss,
        markers=True,
        labels={'x': 'Number of Clusters', 'y': 'WCSS'},
        title="Elbow Method For Optimal Clusters"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Optimal cluster suggestion
    kneedle = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    st.info(f"🧠 Suggested optimal number of clusters: {optimal_k}")

    # Cluster selection
    n_clusters = st.sidebar.slider('Select number of clusters:', min_value=2, max_value=10, value=optimal_k or 3)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    grouped_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Interactive Plotly scatter plot
    fig_plotly = px.scatter(
        grouped_df,
        x='Wellness Center',
        y='Beneficiary Count',
        color=grouped_df['Cluster'].astype(str),
        title='📊 Clustering of Wellness Centers Based on Beneficiaries',
        labels={
            'Wellness Center': 'Wellness Center',
            'Beneficiary Count': 'Number of Beneficiaries',
            'Cluster': 'Cluster'
        },
        hover_data=['Wellness Center', 'Beneficiary Count', 'Cluster']
    )

    fig_plotly.update_layout(
        xaxis_tickangle=45,
        height=700,
        margin=dict(t=50, b=200),
        showlegend=True
    )

    st.plotly_chart(fig_plotly, use_container_width=True)

    with st.expander("🔍 View Cluster Assignments"):
        st.dataframe(grouped_df)

except Exception as e:
    st.error(f"⚠️ Clustering failed: {str(e)}")
