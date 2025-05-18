import os
import pandas as pd
import streamlit as st

# Directory where your CSVs are stored
DATA_DIR = os.path.join(os.getcwd(), "data")

# Define available store types and corresponding CSV filenames
csv_paths = {
    "Kirana":   os.path.join(DATA_DIR, "Kirana_Store_Transactions_v2.csv"),
    "Chemist":  os.path.join(DATA_DIR, "Chemist_Store_Transactions_v2.csv"),
    "Cafe":     os.path.join(DATA_DIR, "Cafe_Store_Transactions_v2.csv"),
    "Clothes":  os.path.join(DATA_DIR, "Clothes_Store_Transactions_v2.csv"),
}

# Load data
data_by_type = {}
for store_type, path in csv_paths.items():
    if os.path.isfile(path):
        data_by_type[store_type] = pd.read_csv(path, parse_dates=["Timestamp"])

# Streamlit UI
st.set_page_config(page_title="PinePulse - Weekly Store Pulse", layout="wide")

st.title("📊 Weekly Store Pulse Report")

selected_type = st.selectbox("Select Store Category", list(data_by_type.keys()))
selected_store = st.text_input("Enter Store Name (case-sensitive):")

if st.button("Generate Store Pulse"):
    df = data_by_type[selected_type]
    store_df = df[df["Store Name"] == selected_store]

    if store_df.empty:
        st.warning("⚠️ No data found for this store.")
    else:
        st.success(f"📈 Showing insights for **{selected_store}** ({selected_type})")
        st.write("Recent Transactions")
        st.dataframe(store_df.sort_values("Timestamp", ascending=False).head(10))

        # Example KPIs
        st.subheader("🧮 Store Metrics")
        st.metric("Total Sales", f"₹{store_df['Total Price'].sum():,.0f}")
        st.metric("Units Sold", int(store_df['Quantity'].sum()))
        st.metric("Unique SKUs Sold", store_df["Item Name"].nunique())

        # Add any additional charts here as needed




