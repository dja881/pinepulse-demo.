import os
import math
import streamlit as st
import pandas as pd
import openai
import altair as alt
import json

# --- INITIALIZE AI CLIENT ---
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- APP CONFIG ---
st.set_page_config(page_title="PinePulse - Interactive Store Dashboard (gpt-4.1-mini)", layout="wide")
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

# --- SIDEBAR CONTROLS ---
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
            # Prepare detailed context with velocity and days_supply
            inv_counts = df.groupby("Product Name")["Quantity"].sum().reset_index().rename(columns={"Quantity":"quantity"})
            top_info = top_df.merge(inv_counts, on="Product Name")
            top_info['velocity'] = top_info['Txn Amount (₹)'] / 20
            top_info['days_supply'] = (top_info['quantity'] / top_info['velocity']).round(1)
            bottom_info = bottom_df.merge(inv_counts, on="Product Name")
            bottom_info['velocity'] = bottom_info['Txn Amount (₹)'] / 20
            bottom_info['days_supply'] = (bottom_info['quantity'] / bottom_info['velocity']).round(1)

            top_context = top_info.to_dict(orient='records')
            bottom_context = bottom_info.to_dict(orient='records')

            # Example few-shot to guide depth and format
            example = {
                "sku": "Parle-G Biscuit (500g)",
                "sales": 3000,
                "quantity": 100,
                "velocity": 150,
                "days_supply": 0.7,
                "recommendations": [
                    "Increase reorder level to 200 units to avoid stockout.",
                    "Run a 10% afternoon promo to boost margin.",
                    "Feature in store entrance display for visibility."
                ]
            }

            sku_prompt = f"""
You are a data-driven retail analyst. For each SKU, you will receive context of sales, inventory, velocity and days_supply.
Use this example as template (valid JSON):
{json.dumps(example, indent=2)}

Now, for the following top SKUs:
{json.dumps(top_context, indent=2)}
Provide 3 distinct, data-backed recommendations per SKU to sustain high sales.

And for the following slow SKUs:
{json.dumps(bottom_context, indent=2)}
Provide 3 distinct, data-backed recommendations per SKU to boost sales.

Return JSON: {{"top_recos": [{{...}}], "bottom_recos": [{{...}}]}}
"""
            with st.spinner("Generating SKU recommendations with gpt-4.1-mini..."):
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role":"system","content":"Output valid JSON only, following the example schema."},
                        {"role":"user","content":sku_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
            try:
                sku_data = json.loads(resp.choices[0].message.content)
            except:
                st.error("Failed to parse SKU recommendations.")
                sku_data = {"top_recos": [], "bottom_recos": []}

            # Render recommendations
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
            # Single Insight
            st.subheader("AI Insight")
            insight_prompt = (
                f"Provide one concise, data-backed insight about {store_name}, referencing sales trends, inventory days_supply, and upcoming weather trends (e.g., 40°C heat)."
            )
            resp_insight = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Output a single-sentence insight."},
                    {"role":"user","content":insight_prompt}
                ],
                temperature=0.3,
                max_tokens=60
            )
            st.info(resp_insight.choices[0].message.content.strip())

            # Forecast metrics
            st.subheader("Next-Month Sales Forecast")
            forecast_prompt = (
                f"Based on transaction, inventory, and external trends (e.g., weather), forecast 3 key category % changes for next month. Return JSON: {{'forecast':[{{'category':'','change':'%'}}]}}"
            )
            resp_forecast = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Output valid JSON with 'forecast'."},
                    {"role":"user","content":forecast_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            try:
                forecast_data = json.loads(resp_forecast.choices[0].message.content)
            except:
                forecast_data = {"forecast": []}
            fcols = st.columns(3)
            for i, f in enumerate(forecast_data.get("forecast", [])):
                fcols[i].metric(f.get("category", ""), f.get("change", ""))

            # AI Nudges
            st.subheader("AI Nudges to Action")
            nudges_prompt = (
                f"Using sales, inventory, days_supply, and external trends, provide 3 actionable nudges for {store_name}. Return JSON: {{'nudges':['...']}}"
            )
            resp_nudges = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Output valid JSON with 'nudges'."},
                    {"role":"user","content":nudges_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            try:
                nudges_data = json.loads(resp_nudges.choices[0].message.content)
            except:
                nudges_data = {"nudges": []}
            for n in nudges_data.get("nudges", []):
                st.write(f"- {n}")

