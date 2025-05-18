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
st.title("PinePulse - Weekly Store Pulse Dashboard")

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

# --- SIDEBAR: CHOOSE DATA SOURCE ---
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Choose Data Source:", ["Use Demo Store Data", "Upload Your Own CSV"])

if data_source == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df_all = pd.read_csv(uploaded_file, parse_dates=["Timestamp"], infer_datetime_format=True)
        store_type = "Uploaded CSV"
    else:
        st.warning("Please upload a file to continue.")
        st.stop()
else:
    store_type = st.sidebar.selectbox("Store Category", list(all_data.keys()))
    df_all = all_data[store_type]

# --- TIME FILTER ---
days = st.sidebar.selectbox("Look at data from past...", [7, 14, 30], index=0)
cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
df_all = df_all[df_all["Timestamp"] >= cutoff]

# --- AUTO-DETECT COLUMNS ---
store_col = next((c for c in df_all.columns if "store" in c.lower()), None)
amount_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["amount","price","total"])), None)
qty_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["remaining","stock","quantity","qty"])), None)
item_col = next((c for c in df_all.columns if any(k in c.lower() for k in ["product name","product","sku"]) and df_all[c].dtype == object), None)

# --- STORE FILTER (only if demo data) ---
if store_col and data_source == "Use Demo Store Data":
    store_name = st.sidebar.selectbox("Store Name", sorted(df_all[store_col].dropna().unique()))
    df_all = df_all[df_all[store_col] == store_name]

# --- SHOW FIRST 30 ROWS ---
st.markdown("### Preview: First 30 Rows of Data")
st.dataframe(df_all.head(30), use_container_width=True)

if st.sidebar.button("Generate Report"):
    df = df_all.loc[:, ~df_all.columns.duplicated()]
    total_sales = df[amount_col].sum()
    num_txn = len(df)
    unique_items = df[item_col].nunique()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales", f"₹{total_sales:,.0f}")
    m2.metric("Transactions", num_txn)
    m3.metric("Unique Products", unique_items)
    st.markdown("---")

    sku_sales = df.groupby(item_col).agg(sales=(amount_col, 'sum')).reset_index()
    n = len(sku_sales)
    top_n = max(1, math.ceil(n * 0.3))
    top_df = sku_sales.nlargest(top_n, 'sales')
    bottom_df = sku_sales.nsmallest(top_n, 'sales')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top {top_n} Movers (Hot-Selling SKUs)")
        chart_top = alt.Chart(top_df).mark_bar().encode(
            x=alt.X("sales:Q", title="Sales"),
            y=alt.Y(f"{item_col}:N", sort='-x', title=None)
        ).properties(height=300)
        st.altair_chart(chart_top, use_container_width=True)
    with col2:
        st.subheader(f"Bottom {top_n} Movers (Cold SKUs)")
        chart_bot = alt.Chart(bottom_df).mark_bar().encode(
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
        ctx['velocity'] = (ctx['sales'] / days).round(1)
        ctx['days_supply'] = ctx.apply(
            lambda r: round(r['quantity'] / r['velocity'], 1) if r['quantity'] and r['velocity'] else None,
            axis=1
        )
        return ctx.to_dict(orient='records')

    top_context = build_ctx(top_df)
    bottom_context = build_ctx(bottom_df)

    product_summary = df.groupby("Product Name").agg(
        total_sales=(amount_col, 'sum'),
        num_txns=('Transaction ID', 'count')
    ).sort_values("total_sales", ascending=False).reset_index()

    payment_summary = df.groupby(["Payment Mode", "Card Type"]).agg(
        total_sales=(amount_col, 'sum'),
        txn_count=('Transaction ID', 'count')
    ).reset_index()

    sku_prompt = f"""
You are a data-driven retail analyst. Follow the example schema:
{json.dumps({
    "sku": "Parle-G Biscuit (500g)",
    "sales": 3000,
    "quantity": 100,
    "velocity": 150,
    "days_supply": 0.7,
    "recommendations": [
        "Current stock may be too high — reduce to ~3 days of supply to lower holding costs.",
        "Schedule a 10% promo during peak hours to boost sales.",
        "Place at checkout for visibility to maximize impulse buys."
    ]
}, indent=2)}

Top SKUs context:
{json.dumps(top_context, indent=2)}

Slow SKUs context:
{json.dumps(bottom_context, indent=2)}

Product-Level Summary:
{json.dumps(product_summary.to_dict(orient="records"), indent=2)}

Payment Summary:
{json.dumps(payment_summary.to_dict(orient="records"), indent=2)}

Return a JSON with these keys: top_recos, bottom_recos, insights, product_insights, payment_insights.
Limit product_insights and payment_insights to 5 key points each. Avoid technical terms like 'days_supply'; use natural phrasing.
"""

    with st.spinner("Generating SKU recommendations and AI insights..."):
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"system","content":"Output valid JSON only."}, {"role":"user","content":sku_prompt}],
            temperature=0.3,
            max_tokens=1200
        )

    try:
        sku_data = json.loads(resp.choices[0].message.content)
    except:
        st.error("Failed to parse SKU recommendations.")
        sku_data = {"top_recos": [], "bottom_recos": [], "insights": [], "product_insights": [], "payment_insights": []}

    with col1:
        st.markdown("**Top SKU Recommendations**")
        for item in sku_data.get("top_recos", []):
            st.write(f"**{item.get('sku', 'Unknown SKU')}**")
            for rec in item.get("recommendations", []):
                st.write(f"- {rec}")

    with col2:
        st.markdown("**Slow SKU Recommendations**")
        for item in sku_data.get("bottom_recos", []):
            st.write(f"**{item.get('sku', 'Unknown SKU')}**")
            for rec in item.get("recommendations", []):
                st.write(f"- {rec}")

    st.markdown("---")
    st.markdown("### AI Forecasts & Strategy Nudges")
    for insight in sku_data.get("insights", [])[:5]:
    st.write(f"- {insight}")


    st.markdown("### Product Insights")
    for insight in sku_data.get("product_insights", [])[:5]:
        st.write(f"- {insight}")

    st.markdown("### Payment Insights")
    for insight in sku_data.get("payment_insights", [])[:5]:
        st.write(f"- {insight}")

