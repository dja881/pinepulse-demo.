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
st.set_page_config(page_title="PinePulse - Interactive Store Dashboard", layout="wide")
st.title("📊 PinePulse - Weekly Store Pulse Dashboard")

# --- DATA PATHS & LOADING ---
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
store_name = None

if store_type:
    df_all = all_data[store_type]
    stores = sorted(df_all["Store Name"].unique().tolist())
    store_name = st.sidebar.selectbox("Store Name", stores)
    if st.sidebar.button("Generate Report"):
        df = df_all[df_all["Store Name"] == store_name]

        # --- BENCHMARK METRICS ---
        total_sales = df["Txn Amount (₹)"].sum()
        num_txn = len(df)
        unique_products = df["Product Name"].nunique()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sales", f"₹{total_sales:,.0f}")
        m2.metric("Transactions", num_txn)
        m3.metric("Unique Products", unique_products)

        st.markdown("---")

        # --- CALCULATE TOP/BOTTOM 30% SKUS ---
        sku_sales = df.groupby("Product Name")["Txn Amount (₹)"].sum().reset_index()
        n = len(sku_sales)
        top_n = max(1, math.ceil(n * 0.3))
        bottom_n = top_n
        top_df = sku_sales.sort_values(by="Txn Amount (₹)", ascending=False).head(top_n)
        bottom_df = sku_sales.sort_values(by="Txn Amount (₹)", ascending=True).head(bottom_n)

        # --- CHARTS FOR TOP/BOTTOM ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 30% Movers (Hot-Selling SKUs)")
            chart_top = alt.Chart(top_df).mark_bar(color="#4CAF50").encode(
                x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                y=alt.Y("Product Name:N", sort="-x", title=None)
            ).properties(height=300)
            st.altair_chart(chart_top, use_container_width=True)
        with c2:
            st.subheader("Bottom 30% Movers (Cold SKUs)")
            chart_bot = alt.Chart(bottom_df).mark_bar(color="#FFA500").encode(
                x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                y=alt.Y("Product Name:N", sort="x", title=None)
            ).properties(height=300)
            st.altair_chart(chart_bot, use_container_width=True)

        # --- AI SKU RECOMMENDATIONS ---
        top_list = top_df['Product Name'].tolist()
        bottom_list = bottom_df['Product Name'].tolist()
        sku_prompt = f"""
You are a retail analyst. For the following hot-selling SKUs: {top_list}, provide a bullet recommendation for each, focusing on restock urgency.
For the following cold SKUs: {bottom_list}, provide a bullet recommendation for each, focusing on discounts or bundling.
Return JSON with keys 'top_recos' and 'bottom_recos', where each is a list of {{'sku':..., 'action':...}}.
"""
        with st.spinner("Generating SKU recommendations..."):
            sku_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You output valid JSON only."},
                    {"role":"user","content":sku_prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
        try:
            sku_data = json.loads(sku_resp.choices[0].message.content)
        except Exception:
            sku_data = {}
            st.error("Failed to parse SKU recommendations.")

        # --- RENDER SKU RECOMMENDATIONS ---
        with c1:
            st.markdown("**Recommendations:**")
            for item in sku_data.get('top_recos', []):
                st.markdown(f"- {item.get('sku')}: {item.get('action')}")
        with c2:
            st.markdown("**Recommendations:**")
            for item in sku_data.get('bottom_recos', []):
                st.markdown(f"- {item.get('sku')}: {item.get('action')}")

        st.markdown("---")

        # --- FOOTFALL PATTERNS & CATEGORY MIX ---
        # (unchanged existing code)

        # --- AI INSIGHTS: Insight, Forecast & Nudges ---
        ai_prompt = f"""
You are a professional data-driven retail analyst. Based on the following:
Store: {store_name} ({store_type}), Transactions: {num_txn}

Provide JSON with keys:
- 'insight': one-sentence summary.
- 'forecast': list of {{'category':..., 'change':'+%-'}}.
- 'nudges': list of top 5 actionable recommendations.
"""
        with st.spinner("Generating AI insight & forecast..."):
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"Output valid JSON only."},
                    {"role":"user","content":ai_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
        try:
            ai_data = json.loads(resp.choices[0].message.content)
        except Exception:
            ai_data = {}
            st.error("Failed to parse AI analysis.")

        if ai_data:
            st.subheader("AI Insight")
            st.write(ai_data.get('insight',''))

            st.subheader("Projected Next-Month Sales Forecast")
            for f in ai_data.get('forecast', []):
                st.metric(f.get('category',''), f.get('change',''))

            st.subheader("AI Nudges to Action This Week")
            for n in ai_data.get('nudges', []):
                st.write(f"- {n}")

