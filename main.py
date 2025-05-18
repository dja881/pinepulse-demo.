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
    # detect columns dynamically
    cols = df_all.columns.tolist()
    # store name col
    store_col = next((c for c in cols if c.lower().startswith("store")), "Store Name")
    # amount col
    amount_col = next((c for c in cols if "amount" in c.lower()), cols[2])
    # quantity col
    quantity_col = next((c for c in cols if "quantity" in c.lower() or "qty" in c.lower()), None)
    # item name col
    item_col = next((c for c in cols if c not in [store_col, amount_col, quantity_col, "Timestamp"] and df_all[c].dtype == object), cols[1])

    stores = sorted(df_all[store_col].unique())
    store_name = st.sidebar.selectbox("Store Name", stores)
    if st.sidebar.button("Generate Report"):
        df = df_all[df_all[store_col] == store_name]

        # --- MAIN METRICS ---
        total_sales = df[amount_col].sum()
        num_txn = len(df)
        unique_items = df[item_col].nunique()
        c_main, c_sidebar = st.columns([3, 1])

        with c_main:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sales", f"₹{total_sales:,.0f}")
            m2.metric("Transactions", num_txn)
            m3.metric("Unique Products", unique_items)
            st.markdown("---")

            # compute sku sales
            sku_sales = df.groupby(item_col)[amount_col].sum().reset_index().rename(columns={amount_col: 'sales'})
            n = len(sku_sales)
            top_n = max(1, math.ceil(n * 0.3))
            top_df = sku_sales.nlargest(top_n, 'sales')
            bottom_df = sku_sales.nsmallest(top_n, 'sales')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Top {top_n} Movers (Hot-Selling)")
                chart_top = alt.Chart(top_df).mark_bar(color="#4CAF50").encode(
                    x=alt.X("sales:Q", title="Sales"),
                    y=alt.Y(f"{item_col}:N", sort='-x', title=None)
                ).properties(height=300)
                st.altair_chart(chart_top, use_container_width=True)
            with col2:
                st.subheader(f"Bottom {top_n} Movers (Slow)")
                chart_bot = alt.Chart(bottom_df).mark_bar(color="#FFA500").encode(
                    x=alt.X("sales:Q", title="Sales"),
                    y=alt.Y(f"{item_col}:N", sort='x', title=None)
                ).properties(height=300)
                st.altair_chart(chart_bot, use_container_width=True)

            # --- AI SKU RECOMMENDATIONS ---
            # prepare context
            inv_counts = None
            if quantity_col and quantity_col in df.columns:
                inv_counts = df.groupby(item_col)[quantity_col].sum().reset_index().rename(columns={quantity_col:'quantity'})
            else:
                inv_counts = pd.DataFrame([{item_col: row[item_col], 'quantity': None} for row in top_df.to_dict('records')])

            def build_context(df_sku):
                ctx = df_sku.merge(inv_counts, on=item_col, how='left')
                ctx['velocity'] = (ctx['sales'] / 20).round(1)
                ctx['days_supply'] = (ctx['quantity'] / ctx['velocity']).round(1) if quantity_col else None
                return ctx.to_dict(orient='records')

            top_context = build_context(top_df)
            bottom_context = build_context(bottom_df)

            # few-shot example
            example = {
                "sku": "Parle-G Biscuit (500g)",
                "sales": 3000,
                "quantity": 100,
                "velocity": 150,
                "days_supply": 0.7,
                "recommendations": [
                    "Increase reorder to 200 units to avoid stockout.",
                    "Run 10% afternoon promo to boost sales.",
                    "Feature in entrance display for visibility."
                ]
            }

            sku_prompt = f"""
You are a data-driven retail analyst. Use this example schema:
{json.dumps(example, indent=2)}

Now for top SKUs:
{json.dumps(top_context, indent=2)}
Provide 3 distinct data-backed "recommendations" per SKU.

And for slow SKUs:
{json.dumps(bottom_context, indent=2)}
Provide 3 distinct data-backed "recommendations" per SKU.

Return JSON: {{"top_recos": [{{...}}], "bottom_recos":[{{...}}]}}
"""
            with st.spinner("Generating SKU recommendations..."):
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role":"system","content":"Output valid JSON only, follow schema."},
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
                f"Provide one concise, data-backed insight about {store_name}, referencing sales trends, inventory days_supply, and external trends like weather."  # user wants specific nod to weather
            )
            resp_insight = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Output a single sentence insight."},
                    {"role":"user","content":insight_prompt}
                ],
                temperature=0.3,
                max_tokens=60
            )
            st.info(resp_insight.choices[0].message.content.strip())

            # forecast metrics
            st.subheader("Next-Month Sales Forecast")
            forecast_prompt = (
                f"Based on sales, inventory, velocity, days_supply, and external trends (e.g., high heat), forecast 3 category-level % changes."  # instruct explicitly
            )
            resp_forecast = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Output valid JSON with key 'forecast'."},
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

            st.subheader("AI Nudges to Action")
            nudges_prompt = (
                f"Using all provided context, generate 3 actionable nudges backed by data and external trends."  # more directive
            )
            resp_nudges = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Output valid JSON with key 'nudges'."},
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

