import os
import streamlit as st
import pandas as pd
import openai
import json

# --- INITIALIZE AI CLIENT ---
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- APP CONFIG ---
st.set_page_config(page_title="PinePulse - AI-Driven Store Insights", layout="wide")
st.title("📊 PinePulse - AI-Driven Store Insights")

# --- DATA PATHS ---
DATA_DIR = os.path.join(os.getcwd(), "data")
csv_paths = {
    "Kirana":   os.path.join(DATA_DIR, "Kirana_Store_Transactions_v2.csv"),
    "Chemist":  os.path.join(DATA_DIR, "Chemist_Store_Transactions_v2.csv"),
    "Cafe":     os.path.join(DATA_DIR, "Cafe_Store_Transactions_v2.csv"),
    "Clothes":  os.path.join(DATA_DIR, "Clothes_Store_Transactions_v2.csv"),
}

# --- CACHEABLE DATA LOADING ---
@st.cache_data
 def load_data(paths):
    data = {}
    for name, path in paths.items():
        if os.path.isfile(path):
            try:
                data[name] = pd.read_csv(path, parse_dates=["Timestamp"])
            except Exception:
                data[name] = pd.read_csv(path)
    return data

data_by_type = load_data(csv_paths)

# --- AI-BASED SCHEMA DETECTION ---
@st.cache_data(ttl=3600)
 def detect_schema(sample_csv: str) -> dict:
    prompt = f"""
You're a smart data analyst. Given this CSV sample, tell me:
1. The column that represents the transaction amount.
2. The column that represents the store name.
3. The column that represents the product/item name.

Respond in JSON with keys: amount, store, product. If not found, return "unknown".

CSV Sample:
{sample_csv}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You label CSV headers based on sample data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=100,
    )
    try:
        schema = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        schema = {"amount": "unknown", "store": "unknown", "product": "unknown"}
    return schema

# --- USER INTERFACE ---
selected_type = st.selectbox("Select Store Category", list(data_by_type.keys()))

if selected_type:
    df = data_by_type[selected_type]
    st.write("**Data Preview (first 5 rows):**")
    st.dataframe(df.head(5))

    # Detect schema dynamically
    sample_csv = df.head(5).to_csv(index=False)
    schema = detect_schema(sample_csv)
    amount_col = schema.get("amount")
    store_col = schema.get("store")
    product_col = schema.get("product")

    # Fallback heuristics if detection fails
    if amount_col == "unknown":
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        amount_col = numeric_cols[0] if numeric_cols else None
    if store_col == "unknown":
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        store_col = object_cols[0] if object_cols else None
    if product_col == "unknown":
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        product_col = object_cols[1] if len(object_cols) > 1 else (object_cols[0] if object_cols else None)

    if not all([amount_col, store_col, product_col]):
        st.error("Unable to determine key columns. Please check your CSV and try again.")
    else:
        st.caption(f"Detected columns → amount: **{amount_col}**, store: **{store_col}**, product: **{product_col}**")
        store_names = df[store_col].dropna().unique().tolist()
        selected_store = st.selectbox("Select Store Name", store_names)

        if st.button("Generate Store Pulse"):
            store_df = df[df[store_col] == selected_store]
            if store_df.empty:
                st.warning("⚠️ No transactions found for this store.")
            else:
                st.subheader(f"Insights for **{selected_store}** ({selected_type})")
                st.write("**Recent Transactions:**")
                st.dataframe(store_df.sort_values(by=store_df.columns[0], ascending=False).head(10))

                st.subheader("Store Metrics")
                st.metric("Total Sales", f"₹{store_df[amount_col].sum():,.0f}")
                st.metric("Transactions", len(store_df))
                st.metric("Unique Products", store_df[product_col].nunique())

                with st.spinner("Generating AI-driven insight..."):
                    insight_prompt = (
                        f"Analyze the transaction data for {selected_store} with {len(store_df)} records "
                        "and write a one-sentence insight."
                    )
                    insight_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a data-driven business analyst."},
                            {"role": "user", "content": insight_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=60,
                    )
                    try:
                        insight_text = insight_response.choices[0].message.content.strip()
                        st.subheader("💡 AI Insight")
                        st.write(insight_text)
                    except Exception:
                        st.error("Failed to parse AI insight.")

