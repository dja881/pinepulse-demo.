import os
import math
import streamlit as st
import pandas as pd
import openai
import altair as alt
import json

# --- INITIALIZE AI CLIENT ---
# Use the gpt-4.1-nano model for AI calls
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- APP CONFIG ---
st.set_page_config(
    page_title="PinePulse - Interactive Store Dashboard (gpt-4.1-nano)",
    layout="wide"
)
st.title("📊 PinePulse - Weekly Store Pulse Dashboard")

# --- DATA LOADING ---
DATA_DIR = os.path.join(os.getcwd(), "data")
csv_paths = {
    "Kirana": os.path.join(DATA_DIR, "Kirana_Store_Transactions_v2.csv"),
    "Chemist": os.path.join(DATA_DIR, "Chemist_Store_Transactions_v2.csv"),
    "Cafe": os.path.join(DATA_DIR, "Cafe_Store_Transactions_v2.csv"),
    "Clothes": os.path.join(DATA_DIR, "Clothes_Store_Transactions_v2.csv"),
}

@st.cache_data
def load_data():
    data = {}
    for name, path in csv_paths.items():
        if os.path.isfile(path):
            df = pd.read_csv(path, parse_dates=["Timestamp"], infer_datetime_format=True)
            data[name] = df
    return data

all_data = load_data()

# --- SIDEBAR CONFIGURATION PANEL ---
st.sidebar.header("Configuration")
store_type = st.sidebar.selectbox("Store Category", list(all_data.keys()))
if store_type:
    df_all = all_data[store_type]
    stores = sorted(df_all["Store Name"].unique())
    store_name = st.sidebar.selectbox("Store Name", stores)
    if st.sidebar.button("Generate Report"):
        df = df_all[df_all["Store Name"] == store_name]

        # --- MAIN METRICS ---
        total_sales = df["Txn Amount (₹)"].sum()
        num_txn = len(df)
        unique_products = df["Product Name"].nunique()
        c_main, c_sidebar = st.columns([3, 1])

        # Main dashboard area
        with c_main:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sales", f"₹{total_sales:,.0f}")
            m2.metric("Transactions", num_txn)
            m3.metric("Unique Products", unique_products)
            st.markdown("---")

            # Compute top/bottom 30% SKUs
            sku_sales = df.groupby("Product Name")["Txn Amount (₹)"].sum().reset_index()
            n = len(sku_sales)
            top_n = max(1, math.ceil(n * 0.3))
            top_df = sku_sales.nlargest(top_n, "Txn Amount (₹)")
            bottom_df = sku_sales.nsmallest(top_n, "Txn Amount (₹)")

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 30% Movers (Hot-Selling SKUs)")
                st.altair_chart(
                    alt.Chart(top_df).mark_bar(color="#4CAF50").encode(
                        x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                        y=alt.Y("Product Name:N", sort="-x", title=None)
                    ).properties(height=300), use_container_width=True
                )
            with col2:
                st.subheader("Bottom 30% Movers (Slow SKUs)")
                st.altair_chart(
                    alt.Chart(bottom_df).mark_bar(color="#FFA500").encode(
                        x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                        y=alt.Y("Product Name:N", sort="x", title=None)
                    ).properties(height=300), use_container_width=True
                )

            # --- AI SKU RECOMMENDATIONS ---
            inv_counts = df.groupby("Product Name")["Quantity"].sum().reset_index().rename(columns={"Quantity":"quantity"})
            top_info = top_df.merge(inv_counts, on="Product Name").to_dict(orient='records')
            bottom_info = bottom_df.merge(inv_counts, on="Product Name").to_dict(orient='records')

            sku_prompt = f"""
You are a data-driven retail analyst. Below are top 30% SKUs and bottom 30% SKUs with their total sales and quantities:

Top SKUs:
{json.dumps(top_info, indent=2)}

Slow SKUs:
{json.dumps(bottom_info, indent=2)}

For each Top SKU, provide 3 distinct recommendations to sustain high sales (e.g., optimal reorder level, targeted promotion, shelf placement).
For each Slow SKU, provide 3 distinct recommendations to boost sales (e.g., bundle offers, discount tiers, repositioning on shelf).

Return a JSON object with keys 'top_recos' and 'bottom_recos'.
"""
            with st.spinner("Generating SKU recommendations with gpt-4.1-nano..."):
                resp = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role":"system","content":"Output valid JSON only."},
                        {"role":"user","content":sku_prompt}
                    ],
                    temperature=0.6,
                    max_tokens=300,
                )
            try:
                sku_data = json.loads(resp.choices[0].message.content)
            except:
                st.error("Failed to parse SKU recommendations.")
                sku_data = {"top_recos": [], "bottom_recos": []}

            with col1:
                st.markdown("**Top SKU Recommendations**")
                for item in sku_data.get("top_recos", []):
                    st.write(f"**{item['sku']}**")
                    for rec in item.get("recommendations", []):
                        st.write(f"- {rec}")
            with col2:
                st.markdown("**Slow SKU Recommendations**")
                for item in sku_data.get("bottom_recos", []):
                    st.write(f"**{item['sku']}**")
                    for rec in item.get("recommendations", []):
                        st.write(f"- {rec}")

            st.markdown("---")

        # --- SIDEBAR AI PANEL: 1-3-3 FORMAT ---
        with c_sidebar:
            st.subheader("AI Insight")
            insight_prompt = (
                f"Provide one concise insight about {store_name} based on {num_txn} transactions over the past 20 days, referencing sales trends and inventory levels."
            )
            resp_insight = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role":"system","content":"Provide a single-sentence data-driven insight."},
                    {"role":"user","content":insight_prompt}
                ],
                temperature=0.7,
                max_tokens=60
            )
            st.info(resp_insight.choices[0].message.content.strip())

            # Forecast metrics
            st.subheader("Next-Month Sales Forecast")
            forecast_prompt = (
                f"Given the transaction and inventory data for {store_name}, forecast 3 category-level % changes for next month. Return JSON: {{'forecast': [{{'category':'','change':'%'}}]}}"
            )
            resp_forecast = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role":"system","content":"Output valid JSON with key 'forecast'."},
                    {"role":"user","content":forecast_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            try:
                forecast_data = json.loads(resp_forecast.choices[0].message.content)
            except:
                forecast_data = {"forecast": []}
            fcols = st.columns(3)
            for i, f in enumerate(forecast_data.get("forecast", [])):
                fcols[i].metric(f.get("category", ""), f.get("change", ""))

            st.subheader("AI Nudges to Action")
            nudges_prompt = (
                f"Using transaction, inventory, and typical external trends (e.g., weather), provide 3 actionable recommendations for {store_name} this week. Return JSON: {{'nudges': ['...']}}"
            )
            resp_nudges = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role":"system","content":"Output valid JSON with key 'nudges'."},
                    {"role":"user","content":nudges_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            try:
                nudges_data = json.loads(resp_nudges.choices[0].message.content)
            except:
                nudges_data = {"nudges": []}
            for n in nudges_data.get("nudges", []):
                st.write(f"- {n}")
