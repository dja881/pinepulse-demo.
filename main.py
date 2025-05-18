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
st.set_page_config(
    page_title="PinePulse - Interactive Store Dashboard",
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

        with c_main:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sales", f"₹{total_sales:,.0f}")
            m2.metric("Transactions", num_txn)
            m3.metric("Unique Products", unique_products)
            st.markdown("---")

            # Top/Bottom SKU charts
            sku_sales = df.groupby("Product Name")["Txn Amount (₹)"].sum().reset_index()
            n = len(sku_sales)
            top_n = max(1, math.ceil(n * 0.3))
            top_df = sku_sales.nlargest(top_n, "Txn Amount (₹)")
            bottom_df = sku_sales.nsmallest(top_n, "Txn Amount (₹)")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 30% Movers (Hot-Selling SKUs)")
                chart_top = alt.Chart(top_df).mark_bar(color="#4CAF50").encode(
                    x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                    y=alt.Y("Product Name:N", sort="-x", title=None)
                ).properties(height=300)
                st.altair_chart(chart_top, use_container_width=True)
            with col2:
                st.subheader("Bottom 30% Movers (Cold SKUs)")
                chart_bot = alt.Chart(bottom_df).mark_bar(color="#FFA500").encode(
                    x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                    y=alt.Y("Product Name:N", sort="x", title=None)
                ).properties(height=300)
                st.altair_chart(chart_bot, use_container_width=True)

            # --- AI SKU RECOMMENDATIONS ---
            # Build detailed context from our top/bottom 30% dataframes
            top_info = (
                top_df
                .rename(columns={'Txn Amount (₹)': 'sales'})
                [['Product Name', 'sales']]
                .to_dict(orient='records')
            )
            bottom_info = (
                bottom_df
                .rename(columns={'Txn Amount (₹)': 'sales'})
                [['Product Name', 'sales']]
                .to_dict(orient='records')
            )

            sku_prompt = f"""
You are a data-driven retail analyst. Here are our hot-selling SKUs (top 30%) with their total sales over the period:
{json.dumps(top_info, indent=2)}

And here are our cold SKUs (bottom 30%) with their total sales:
{json.dumps(bottom_info, indent=2)}

For each hot-selling SKU, recommend exactly one action to sustain and further optimize its performance (e.g., adjust reorder quantity, run a targeted promo, etc.).
For each cold SKU, recommend exactly one action to boost its sales (e.g., bundling strategy, discount tiers, repositioning on shelf).

Return a JSON object with two keys:
  "top_recos": [{"sku": string, "action": string}, …],
  "bottom_recos": [{"sku": string, "action": string}, …]
"""

            with st.spinner("Generating SKU recommendations..."):
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You output valid JSON only."},
                        {"role": "user",   "content": sku_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=200,
                )

            try:
                sku_data = json.loads(resp.choices[0].message.content)
            except Exception:
                st.error("Failed to parse SKU recommendations.")
                sku_data = {"top_recos": [], "bottom_recos": []}

            # Render recommendations under charts
            with col1:
                st.markdown("**Hot-Mover Actions**")
                for item in sku_data.get("top_recos", []):
                    st.markdown(f"- **{item['sku']}**: {item['action']}")
            with col2:
                st.markdown("**Cold-SKU Actions**")
                for item in sku_data.get("bottom_recos", []):
                    st.markdown(f"- **{item['sku']}**: {item['action']}")

            st.markdown("---")

            # Footfall patterns
            df["Weekday"] = df["Timestamp"].dt.day_name()
            df["Hour"] = df["Timestamp"].dt.hour
            by_day = df.groupby("Weekday").size().reset_index(name="count")
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            by_day["Weekday"] = pd.Categorical(by_day["Weekday"], categories=order, ordered=True)
            by_day = by_day.sort_values("Weekday")
            by_hour = df.groupby("Hour").size().reset_index(name="count")
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Transactions by Day")
                st.altair_chart(
                    alt.Chart(by_day).mark_line(point=True).encode(
                        x="Weekday:N",
                        y="count:Q"
                    ).properties(height=250),
                    use_container_width=True
                )
            with c4:
                st.subheader("Transactions by Hour")
                st.altair_chart(
                    alt.Chart(by_hour).mark_line(point=True).encode(
                        x="Hour:O",
                        y="count:Q"
                    ).properties(height=250),
                    use_container_width=True
                )

            st.markdown("---")
            # Category mix
            cat_mix = df.groupby("Product Category")["Txn Amount (₹)"].sum().reset_index()
            st.subheader("Category Revenue Mix")
            st.altair_chart(
                alt.Chart(cat_mix).mark_arc().encode(
                    theta="Txn Amount (₹):Q",
                    color="Product Category:N"
                ).properties(height=300),
                use_container_width=True
            )

        # --- SIDEBAR AI PANEL: 1-3-3 FORMAT ---
        with c_sidebar:
            st.subheader("AI Insight")
            insight_prompt = (
                f"Provide one concise insight about {store_name} based on {num_txn} transactions over the past 20 days."
            )
            resp_insight = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional retail analyst. Provide a single-sentence insight."},
                    {"role": "user", "content": insight_prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            insight = resp_insight.choices[0].message.content.strip()
            st.info(insight)

            st.subheader("Next-Month Sales Forecast")
            forecast_prompt = (
                f"Give 3 bullet forecast percentage changes for key categories at {store_name}. Return JSON: {json.dumps({'views': []})}"
            )
            resp_forecast = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"Output valid JSON with 3 forecast items."},
                    {"role":"user","content":forecast_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            try:
                forecast_data = json.loads(resp_forecast.choices[0].message.content)
            except:
                forecast_data = {"views": []}
            cols = st.columns(3)
            for idx, item in enumerate(forecast_data.get("views", [])):
                cols[idx].metric(item.get("category", ""), item.get("change", ""))

            st.subheader("AI Nudges to Action")
            nudges_prompt = (
                f"List 3 top actionable recommendations for {store_name} this week. Return JSON: {json.dumps({'nudges': []})}"
            )
            resp_nudges = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"Output valid JSON with 3 nudges."},
                    {"role":"user","content":nudges_prompt}
                ],
                temperature=0.7,
                max_tokens=80
            )
            try:
                nudges_data = json.loads(resp_nudges.choices[0].message.content)
            except:
                nudges_data = {"nudges": []}
            for n in nudges_data.get("nudges", []):
                st.write(f"- {n}")

