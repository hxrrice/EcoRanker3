
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="COBRA: Big Data MCDM for Sustainability", layout="wide")

# --- Title and Description ---
st.title("🌱 COBRA: Big Data MCDM for Sustainability and Innovation Impact")
st.markdown("""
COBRA is a decision support tool using the *COBRA method* to rank alternatives
(such as innovations, regions, or projects) based on *sustainability and impact* criteria.

Built with *Streamlit* and powered by *Big Data and MCDM*, this system helps 
evaluate performance across multiple dimensions with clarity and transparency.
""")

# --- Load Example Dataset ---
def load_example_data():
    data = {
        "Project": ["Project A", "Project B", "Project C", "Project D"],
        "Energy Efficiency": [85, 90, 78, 88],
        "Carbon Reduction": [70, 65, 85, 80],
        "Innovation Index": [90, 80, 88, 75],
        "Scalability": [75, 85, 70, 80]
    }
    return pd.DataFrame(data)

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload Decision Matrix (CSV or Excel)", type=["csv", "xlsx"])
use_example = st.checkbox("Use example sustainability dataset", value=not uploaded_file)

# --- Read Data ---
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
elif use_example:
    df = load_example_data()
else:
    st.warning("Please upload a dataset or select the example.")
    st.stop()

# --- Validate Data ---
if df.shape[1] < 3:
    st.error("Dataset must include at least 1 identifier column and 2+ numeric criteria.")
    st.stop()

# --- Extract data ---
alternative_names = df.iloc[:, 0]
criteria_matrix = df.iloc[:, 1:]

# --- Input Weights and Impacts ---
st.subheader("⚖️ Criteria Weights and Impacts")

num_criteria = criteria_matrix.shape[1]
default_weights = [round(1 / num_criteria, 2)] * num_criteria

weights_input = st.text_input("Enter weights (comma-separated, must sum to 1):", ",".join(map(str, default_weights)))
impacts_input = st.text_input("Enter impacts (+ for benefit, - for cost):", ",".join(['+'] * num_criteria))

# --- Parse weights and impacts ---
try:
    weights = list(map(float, weights_input.split(',')))
    impacts = [i.strip() for i in impacts_input.split(',')]

    if len(weights) != num_criteria or len(impacts) != num_criteria:
        st.error("Weights and impacts must match number of criteria.")
        st.stop()
    if not np.isclose(sum(weights), 1.0):
        st.error("Weights must sum to 1.")
        st.stop()
    if not all(i in ['+', '-'] for i in impacts):
        st.error("Impacts must be '+' or '-' only.")
        st.stop()
except:
    st.error("Error parsing weights or impacts.")
    st.stop()

# --- Normalize matrix (Vector normalization) ---
normalized_matrix = criteria_matrix / np.sqrt((criteria_matrix ** 2).sum())
st.subheader("📊 Normalized Decision Matrix")
st.dataframe(pd.concat([alternative_names, normalized_matrix], axis=1))

# --- Apply Weights ---
weighted_matrix = normalized_matrix * weights
st.subheader("📌 Weighted Normalized Matrix")
st.dataframe(pd.concat([alternative_names, weighted_matrix], axis=1))

# --- Determine Ideal (PIS), Negative Ideal (NIS), and Average Solution (AS) ---
PIS = weighted_matrix.max(axis=0)
NIS = weighted_matrix.min(axis=0)
AS = weighted_matrix.mean(axis=0)

# Adjust for cost criteria (minimize instead of maximize)
for i, impact in enumerate(impacts):
    if impact == '-':
        PIS[i] = weighted_matrix.min(axis=0)[i]
        NIS[i] = weighted_matrix.max(axis=0)[i]

# --- Compute Distances ---
distance_to_PIS = np.sqrt(((weighted_matrix - PIS) ** 2).sum(axis=1))
distance_to_NIS = np.sqrt(((weighted_matrix - NIS) ** 2).sum(axis=1))
distance_to_AS = np.sqrt(((weighted_matrix - AS) ** 2).sum(axis=1))

# --- Compute Closeness Scores ---
closeness_scores_PIS = distance_to_NIS / (distance_to_PIS + distance_to_NIS)
closeness_scores_AS = distance_to_AS / (distance_to_PIS + distance_to_AS)

# --- Rank Alternatives ---
ranking = pd.DataFrame({
    "Alternative": alternative_names,
    "Closeness Score (PIS)": closeness_scores_PIS,
    "Rank (PIS)": closeness_scores_PIS.rank(ascending=False).astype(int),
    "Closeness Score (AS)": closeness_scores_AS,
    "Rank (AS)": closeness_scores_AS.rank(ascending=False).astype(int)
}).sort_values(by="Rank (PIS)", ascending=True)

# --- Highlight Top Rank ---
st.subheader("🏆 Final Ranking (COBRA)")
def highlight_top(row):
    return ['background-color: lightgreen; font-weight: bold' if row['Rank (PIS)'] == 1 else '' for _ in row]
st.dataframe(ranking.style.apply(highlight_top, axis=1), use_container_width=True)

# --- Download Results ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='COBRA Results')
    return output.getvalue()

excel_data = to_excel(ranking)
st.download_button("📥 Download Results as Excel", data=excel_data, file_name="cobramethod_results.xlsx")

# --- Footer ---
st.markdown("---")
st.markdown("Created with 💚 for the theme Celebrating Innovation, Commercialisation, and Publication. Powered by COBRA, Big Data & Streamlit.")
