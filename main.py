import os
import streamlit as st
import pandas as pd
import openai

# --- SETUP API KEY ---
openai.api_key = st.secrets["openai"]["api_key"]

# --- DATA LOADING SETUP ---
DATA_DIR = os.path.join(os.getcwd(), "data")

csv_paths = {
    "Kirana":   os.path.join(DATA_DIR, "Kirana_Store_Transactions_v2.csv"),
    "Chemist":  os.path.join(DATA_DIR, "Chemist_Store_Transactions_v2.csv"),
    "Cafe":     os.path.join(DATA_DIR, "Cafe_Store_Transactions_v2.csv"),
    "Clothes":  os.path.join(DATA_DIR, "Clothes_Store_Transactions_v2.csv"),
}

data_by_type = {}
for store_type, path in csv_paths.items():
    if os.path.isfile(path):
        data_by_type[store_type] = pd.read_csv(path, parse_dates=["Timestamp"])

# --- STREAMLIT UI ---
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

        st.subheader("🧮 Store Metrics")
        st.metric("Total Sales", f"₹{store_df['Total Price'].sum():,.0f}")
        st.metric("Units Sold", int(store_df['Quantity'].sum()))
        st.metric("Unique SKUs Sold", store_df["Item Name"].nunique())

        # Optional: Use OpenAI to generate a fun summary
        with st.spinner("Asking OpenAI for a store summary..."):
            prompt = f"Write a quick one-line insight about the sales pattern at {selected_store} based on {selected_type} store data with {len(store_df)} rows of transactions."
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a witty business analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=50
                )
                insight = response.choices[0].message.content.strip()
                st.subheader("💡 AI Insight")
                st.write(insight)
            except Exception as e:
                st.error(f"Failed to generate AI insight: {e}")



