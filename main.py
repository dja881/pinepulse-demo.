import os
import streamlit as st
import pandas as pd
import openai
import json

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
st.title("\U0001F4CA Weekly Store Pulse Report")

selected_type = st.selectbox("Select Store Category", list(data_by_type.keys()))

if selected_type:
    df = data_by_type[selected_type]
    sample = df.head(5).to_csv(index=False)

    prompt = f"""
You're a smart data analyst. Based on this sample CSV, tell me:
1. Which column is the transaction amount?
2. Which column is the store name?
3. Which column describes the item or product?

Return a JSON with keys: amount, store, product. If not found, return \"unknown\".

CSV Sample:
{sample}
"""

    try:
        schema_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You analyze and label CSV headers based on sample data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        schema = json.loads(schema_response.choices[0].message.content)
        amount_col = schema.get("amount", "unknown")
        store_col = schema.get("store", "unknown")
        product_col = schema.get("product", "unknown")

        if "unknown" in [amount_col, store_col, product_col]:
            st.error("❌ GPT couldn't confidently detect all key columns. Please check your CSV.")
        else:
            store_names = df[store_col].unique()
            selected_store = st.selectbox("Select Store Name", store_names)

            if st.button("Generate Store Pulse"):
                store_df = df[df[store_col] == selected_store]

                if store_df.empty:
                    st.warning("⚠️ No data found for this store.")
                else:
                    st.success(f"📈 Showing insights for **{selected_store}** ({selected_type})")
                    st.write("Recent Transactions")
                    st.dataframe(store_df.sort_values("Timestamp", ascending=False).head(10))

                    st.subheader("\U0001F9EE Store Metrics")
                    st.metric("Total Sales", f"₹{store_df[amount_col].sum():,.0f}")
                    st.metric("Transactions", len(store_df))
                    st.metric("Categories Sold", store_df[product_col].nunique())

                    with st.spinner("Asking OpenAI for a store insight..."):
                        insight_prompt = f"Write a quick one-line insight about {selected_store} using {len(store_df)} rows of transaction data."
                        insight_response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You're a witty retail data analyst."},
                                {"role": "user", "content": insight_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=50
                        )
                        st.subheader("\U0001F4A1 AI Insight")
                        st.write(insight_response.choices[0].message.content.strip())

    except Exception as e:
        st.error(f"Something went wrong with schema detection: {e}")


