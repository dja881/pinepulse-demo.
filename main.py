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

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Choose Data Source:", ["Use Demo Store Data", "Upload Your Own CSV"])

if data_source == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df_all = pd.read_csv(uploaded_file, parse_dates=["Timestamp"], infer_datetime_format=True)
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
def detect_col(preferred, columns):
    for option in preferred:
        match = next((c for c in columns if option.lower() in c.lower()), None)
        if match:
            return match
    return None

store_col = detect_col(["Store Name"], df_all.columns)
amount_col = detect_col(["Total Amount", "Price per Unit", "Amount"], df_all.columns)
qty_col = detect_col(["Stock Remaining", "Quantity Sold", "Quantity"], df_all.columns)
item_col = detect_col(["Product Name", "SKU"], df_all.columns)

# --- DEMO STORE FILTER ---
if store_col and data_source == "Use Demo Store Data":
    store_name = st.sidebar.selectbox("Store Name", sorted(df_all[store_col].dropna().unique()))
    df_all = df_all[df_all[store_col] == store_name]

# --- PREVIEW DATA ---
st.markdown("### Preview: First 30 Rows of Data")
st.dataframe(df_all.head(30), use_container_width=True)

# --- GENERATE REPORT BUTTON ---
if st.sidebar.button("Generate Report"):
    df = df_all.copy()
    total_sales = df[amount_col].sum()
    num_txn = len(df)
    unique_items = df[item_col].nunique()

    # --- KPIs ---
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"₹{total_sales:,.0f}")
    k2.metric("Transactions", num_txn)
    k3.metric("Unique Products", unique_items)
    st.markdown("---")

    # --- SKU PERFORMANCE ---
    sku_sales = df.groupby(item_col).agg(sales=(amount_col, 'sum')).reset_index()
    top_n = max(1, math.ceil(len(sku_sales) * 0.3))
    top_df = sku_sales.nlargest(top_n, 'sales')
    bottom_df = sku_sales.nsmallest(top_n, 'sales')

    # --- INVENTORY CONTEXT ---
    if qty_col:
        inv = df.groupby(item_col)[qty_col].sum().reset_index().rename(columns={qty_col: 'quantity'})
    else:
        inv = pd.DataFrame({item_col: top_df[item_col], 'quantity': [None]*len(top_df)})

    def build_ctx(base_df):
        ctx = base_df.merge(inv, on=item_col, how='left')
        ctx['velocity'] = (ctx['sales'] / days).round(1)
        ctx['days_supply'] = ctx.apply(
            lambda r: round(r['quantity'] / r['velocity'], 1) if r['quantity'] and r['velocity'] else None,
            axis=1
        )
        return ctx.to_dict(orient='records')

    top_ctx = build_ctx(top_df)
    bot_ctx = build_ctx(bottom_df)

    # --- SUMMARIES FOR AI PROMPT ---
    product_summary = df.groupby(item_col).agg(
        total_sales=(amount_col, 'sum'),
        count=('Quantity Sold', 'count')
    ).sort_values('total_sales', ascending=False).reset_index()

    payment_summary = df.groupby(['Payment Mode', 'Card Type']).agg(
        total_sales=(amount_col, 'sum'),
        count=(amount_col, 'count')
    ).reset_index()

    category_summary = df.groupby('Category').agg(
        total_sales=(amount_col, 'sum')
    ).sort_values('total_sales', ascending=False).reset_index()

    # --- DISPLAY CHARTS ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top {top_n} Movers (Hot-Selling SKUs)")
        chart1 = alt.Chart(top_df).mark_bar().encode(
            x='sales:Q',
            y=alt.Y(f'{item_col}:N', sort='-x')
        ).properties(height=300)
        st.altair_chart(chart1, use_container_width=True)
    with c2:
        st.subheader(f"Bottom {top_n} Movers (Cold SKUs)")
        chart2 = alt.Chart(bottom_df).mark_bar().encode(
            x='sales:Q',
            y=alt.Y(f'{item_col}:N', sort='x')
        ).properties(height=300)
        st.altair_chart(chart2, use_container_width=True)

    st.markdown("---")

    # --- AI INSIGHTS ---
    prompt = f"""
You are a data-driven retail analyst. Return exactly 3 insights each for categories, products, and overall strategy (include payments).

Category Summary:
{json.dumps(category_summary.to_dict(orient='records')[:5], indent=2)}

Product Summary:
{json.dumps(product_summary.to_dict(orient='records')[:5], indent=2)}

Payment Summary:
{json.dumps(payment_summary.to_dict(orient='records')[:5], indent=2)}

Top SKU Context:
{json.dumps(top_ctx[:5], indent=2)}

Slow SKU Context:
{json.dumps(bot_ctx[:5], indent=2)}
"""
    resp = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[{'role':'system','content':'Output valid JSON only.'}, {'role':'user','content':prompt}],
        temperature=0.3,
        max_tokens=1200
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except:
        st.error('Failed to parse SKU recommendations.')
        data = {'category_insights':[], 'product_insights':[], 'insights':[]}

    st.markdown("### Category Insights")
    for line in data.get('category_insights', [])[:3]: st.markdown(f"- {line}")

    st.markdown("### Product Insights")
    for line in data.get('product_insights', [])[:3]: st.markdown(f"- {line}")

    st.markdown("### AI Forecasts & Strategy Nudges")
    for line in data.get('insights', [])[:3]: st.markdown(f"- {line}")

    st.markdown("### Category-Level Summary Table")
    st.dataframe(category_summary, use_container_width=True)

