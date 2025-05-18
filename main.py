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
st.markdown("""
    <style>
    h1 { font-size: 2.2rem; margin-bottom: 1rem; }
    h3 { color: #888; margin-top: 2rem; }
    .section { margin-top: 2rem; margin-bottom: 2rem; }
    .recos { padding-left: 1rem; }
    </style>
""", unsafe_allow_html=True)

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
            data[name] = pd.read_csv(path, parse_dates=["Timestamp"], infer_datetime_format=True)
    return data

all_data = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")
store_type = st.sidebar.selectbox("Store Category", list(all_data.keys()))
if store_type:
    df_all = all_data[store_type]
    store_col = next((c for c in df_all.columns if "store" in c.lower()), None)
    amount_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["amount","price","total"])), None)
    qty_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["remaining","stock","quantity","qty"])), None)
    item_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["product name","product","sku"]) and df_all[c].dtype == object), None)

    store_name = st.sidebar.selectbox("Store Name", sorted(df_all[store_col].dropna().unique()))
    if st.sidebar.button("Generate Report"):
        df = df_all[df_all[store_col] == store_name]

        total_sales = df[amount_col].sum()
        num_txn = len(df)
        unique_items = df[item_col].nunique()

        st.markdown("### Summary Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sales", f"₹{total_sales:,.0f}")
        m2.metric("Transactions", num_txn)
        m3.metric("Unique Products", unique_items)
        st.divider()

        sku_sales = df.groupby(item_col).agg(sales=(amount_col, 'sum')).reset_index()
        n = len(sku_sales)
        top_n = max(1, math.ceil(n * 0.3))
        top_df = sku_sales.nlargest(top_n, 'sales')
        bottom_df = sku_sales.nsmallest(top_n, 'sales')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top-Selling Products")
            chart_top = alt.Chart(top_df).mark_bar(color="#4CAF50").encode(
                x=alt.X("sales:Q", title="Sales"),
                y=alt.Y(f"{item_col}:N", sort='-x', title=None)
            ).properties(height=300)
            st.altair_chart(chart_top, use_container_width=True)
        with col2:
            st.markdown("### Low-Selling Products")
            chart_bot = alt.Chart(bottom_df).mark_bar(color="#FFA500").encode(
                x=alt.X("sales:Q", title="Sales"),
                y=alt.Y(f"{item_col}:N", sort='x', title=None)
            ).properties(height=300)
            st.altair_chart(chart_bot, use_container_width=True)

        if qty_col:
            inv = df.groupby(item_col)[qty_col].sum().reset_index().rename(columns={qty_col:'quantity'})
        else:
            inv = pd.DataFrame({item_col: top_df[item_col], 'quantity': [None]*len(top_df)})

        def build_ctx(df_sku):
            ctx = df_sku.merge(inv, on=item_col, how='left')
            ctx['velocity'] = (ctx['sales'] / 20).round(1)
            ctx['days_supply'] = ctx.apply(lambda r: round(r['quantity']/r['velocity'],1) if r['quantity'] and r['velocity'] else None, axis=1)
            return ctx.to_dict(orient='records')

        top_context = build_ctx(top_df)
        bottom_context = build_ctx(bottom_df)

        example = {
            "sku": "Parle-G Biscuit (500g)",
            "sales": 3000,
            "quantity": 100,
            "velocity": 150,
            "days_supply": 0.7,
            "recommendations": [
                "Set reorder level to 200 to avoid stockout.",
                "Schedule a 10% promo during peak hours.",
                "Place at checkout for visibility."
            ]
        }

        sku_prompt = f"""
You are a data-driven retail analyst. Follow the example schema:
{json.dumps(example, indent=2)}

Now top SKUs context:
{json.dumps(top_context, indent=2)}
Provide exactly 3 data-backed "recommendations" per SKU.

Slow SKUs context:
{json.dumps(bottom_context, indent=2)}
Provide exactly 3 data-backed "recommendations" per SKU.

Then give 4 AI insights about:
1. Trends in sales and demand patterns,
2. External signals (e.g. weather, festivals),
3. Inventory risks or opportunities,
4. Recommendations for next month’s prep.

Return JSON: {{"top_recos": [...], "bottom_recos": [...], "insights": [...]}}
"""
        with st.spinner("Generating Recommendations & Insights..."):
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role":"system","content":"Output valid JSON only."}, {"role":"user","content":sku_prompt}],
                temperature=0.3,
                max_tokens=800
            )
        try:
            sku_data = json.loads(resp.choices[0].message.content)
        except:
            st.error("Failed to parse response.")
            sku_data = {"top_recos": [], "bottom_recos": [], "insights": []}

        with col1:
            st.markdown("### Recommendations for Top-Sellers")
            for item in sku_data.get("top_recos", []):
                st.write(f"**{item['sku']}**")
                for rec in item.get("recommendations", []): st.markdown(f"- {rec}")
        with col2:
            st.markdown("### Recommendations for Cold-Movers")
            for item in sku_data.get("bottom_recos", []):
                st.write(f"**{item['sku']}**")
                for rec in item.get("recommendations", []): st.markdown(f"- {rec}")

        st.divider()
        st.markdown("### 🔍 AI Forecasts & Nudges")
        for insight in sku_data.get("insights", []):
            st.markdown(f"- {insight}")


