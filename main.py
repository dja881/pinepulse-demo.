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
    # dynamic column detection
    store_col = next((c for c in df_all.columns if "store" in c.lower()), None)
    amount_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["amount","price","total"])), None)
    qty_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["remaining","stock","quantity","qty"])), None)
    item_col = next((c for c in df_all.columns if c not in [store_col, amount_col, qty_col, "Timestamp"] and df_all[c].dtype == object), None)

    store_name = st.sidebar.selectbox("Store Name", sorted(df_all[store_col].dropna().unique()))
    if st.sidebar.button("Generate Report"):
        df = df_all[df_all[store_col] == store_name]

        # --- MAIN METRICS ---
        total_sales = df[amount_col].sum()
        num_txn = len(df)
        unique_items = df[item_col].nunique()
        c_main, c_sidebar = st.columns([3,1])

        # Main dashboard
        with c_main:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sales", f"₹{total_sales:,.0f}")
            m2.metric("Transactions", num_txn)
            m3.metric("Unique Products", unique_items)
            st.markdown("---")

            # compute sales by item (SKU level)
            sku_sales = (
                df.groupby(item_col)
                  .agg(sales=(amount_col, 'sum'))
                  .reset_index()
            )
            n = len(sku_sales)
            top_n = max(1, math.ceil(n * 0.3))
            top_df = sku_sales.nlargest(top_n, 'sales')
            bottom_df = sku_sales.nsmallest(top_n, 'sales')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Top {top_n} Movers (Hot-Selling SKUs)")
                chart_top = alt.Chart(top_df).mark_bar(color="#4CAF50").encode(
                    x=alt.X("sales:Q", title="Sales"),
                    y=alt.Y(f"{item_col}:N", sort='-x', title=None)
                ).properties(height=300)
                st.altair_chart(chart_top, use_container_width=True)
            with col2:
                st.subheader(f"Bottom {top_n} Movers (Cold SKUs)")
                chart_bot = alt.Chart(bottom_df).mark_bar(color="#FFA500").encode(
                    x=alt.X("sales:Q", title="Sales"),
                    y=alt.Y(f"{item_col}:N", sort='x', title=None)
                ).properties(height=300)
                st.altair_chart(chart_bot, use_container_width=True)

            # prepare context for AI
            if qty_col:
                inv = (
                    df.groupby(item_col)[qty_col]
                      .sum()
                      .reset_index()
                      .rename(columns={qty_col:'quantity'})
                )
            else:
                inv = pd.DataFrame({item_col: top_df[item_col], 'quantity': [None]*len(top_df)})

            def build_ctx(df_sku):
                ctx = df_sku.merge(inv, on=item_col, how='left')
                ctx['velocity'] = (ctx['sales'] / 20).round(1)
                ctx['days_supply'] = ctx.apply(lambda r: round(r['quantity']/r['velocity'],1) if r['quantity'] and r['velocity'] else None, axis=1)
                return ctx.to_dict(orient='records')

            top_context = build_ctx(top_df)
            bottom_context = build_ctx(bottom_df)

            # few-shot example
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

Return JSON: {{"top_recos": [...], "bottom_recos": [...]}}
"""
            with st.spinner("Generating SKU recommendations..."):
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role":"system","content":"Output valid JSON only."}, {"role":"user","content":sku_prompt}],
                    temperature=0.3,
                    max_tokens=500
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
                    for rec in item.get("recommendations", []): st.write(f"- {rec}")
            with col2:
                st.markdown("**Slow SKU Recommendations**")
                for item in sku_data.get("bottom_recos", []):
                    st.write(f"**{item['sku']}**")
                    for rec in item.get("recommendations", []): st.write(f"- {rec}")

            st.markdown("---")


