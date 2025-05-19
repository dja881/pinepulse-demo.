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
            data[name] = pd.read_csv(path, parse_dates=["Timestamp"])
    return data

all_data = load_data()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Choose Data Source:", ["Use Demo Store Data", "Upload Your Own CSV"])

if data_source == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df_all = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
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

# --- DETECT COLUMNS ---
def detect_col(preferred, cols):
    for p in preferred:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return None

store_col = detect_col(["Store Name"], df_all.columns)
amount_col = detect_col(["Total Amount", "Price per Unit"], df_all.columns)
qty_col = detect_col(["Stock Remaining", "Quantity Sold"], df_all.columns)
item_col = detect_col(["Product Name", "SKU"], df_all.columns)

# --- DEMO STORE FILTER ---
if store_col and data_source == "Use Demo Store Data":
    name = st.sidebar.selectbox("Store Name", sorted(df_all[store_col].unique()))
    df_all = df_all[df_all[store_col] == name]

# --- PREVIEW DATA ---
st.markdown("### Preview: First 30 Rows of Data")
st.dataframe(df_all.head(30), use_container_width=True)

if st.sidebar.button("Generate Report"):
    df = df_all.copy()
    # KPIs
    total_sales = df[amount_col].sum()
    num_txn = len(df)
    unique_items = df[item_col].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"₹{total_sales:,.0f}")
    c2.metric("Transactions", num_txn)
    c3.metric("Unique Products", unique_items)
    st.markdown("---")

    # Summaries
    sku_sales = df.groupby(item_col).agg(sales=(amount_col, 'sum')).reset_index()
    top_n = max(1, math.ceil(len(sku_sales)*0.3))
    top_df = sku_sales.nlargest(top_n, 'sales')
    bottom_df = sku_sales.nsmallest(top_n, 'sales')
    category_summary = df.groupby('Category').agg(total_sales=(amount_col, 'sum')).reset_index()

    # Context for AI
    inv = df.groupby(item_col)[qty_col].sum().reset_index().rename(columns={qty_col:'quantity'}) if qty_col else pd.DataFrame({item_col:top_df[item_col],'quantity':[None]*len(top_df)})
    def build_ctx(df_sku):
        ctx = df_sku.merge(inv, on=item_col, how='left')
        ctx['velocity'] = (ctx['sales']/days).round(1)
        ctx['days_supply'] = ctx.apply(lambda r: round(r['quantity']/r['velocity'],1) if r['quantity'] and r['velocity'] else None, axis=1)
        return ctx.to_dict('records')
    top_ctx = build_ctx(top_df)
    bot_ctx = build_ctx(bottom_df)

        # --- EXAMPLE-DRIVEN AI PROMPT ---
    example_json = {
        "category_insights": [
            "Snacks are leading in sales, indicating strong consumer preference.",
            "Dairy category has high stock but moderate sales, posing a wastage risk.",
            "Household items have seen a recent spike — likely due to end-of-month cleaning habits."
        ],
        "product_insights": [
            "Parle-G has the highest repeat purchase rate — move it closer to billing counter.",
            "Maggi is underperforming despite good stock — push through shelf positioning.",
            "Amul Milk is consistently bought in singles — explore combo with bread."
        ],
        "insights": [
            "Demand is peaking on weekends — staffing should match footfall trends.",
            "UPI is used for 60%+ transactions — enable QR-based loyalty incentives.",
            "Inventory turnover is faster than restocking — avoid stockouts for top 3 SKUs."
        ]
    }

    prompt = f"""
You are a data-driven retail analyst. Follow the format and tone of the example below. Return valid JSON with exactly these keys: category_insights, product_insights, insights. Each list must have 3 bullet points. Include at least one payment-related comment under 'insights'.

Example:
{json.dumps(example_json, indent=2)}

Category Summary:
{json.dumps(category_summary.to_dict('records')[:5], indent=2)}

Product Summary:
{json.dumps(top_df.to_dict('records')[:5], indent=2)}

Payment Summary:
{json.dumps(df.groupby(['Payment Mode','Card Type']).agg(total_sales=(amount_col,'sum')).reset_index().to_dict('records')[:5], indent=2)}

Top SKU Context:
{json.dumps(top_ctx[:5], indent=2)}

Slow SKU Context:
{json.dumps(bot_ctx[:5], indent=2)}
"""
    resp = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[{'role':'system','content':'Output valid JSON only.'},{'role':'user','content':prompt}],
        temperature=0.3,
        max_tokens=1200
    )
    # Debug: show raw AI response
    st.text(resp.choices[0].message.content)
    try:
        insights = json.loads(resp.choices[0].message.content)
        # Fallback: if category_insights or product_insights are empty, split from insights
        all_ins = insights.get('insights', [])
        if not insights.get('category_insights') and all_ins:
            insights['category_insights'] = all_ins[:3]
        if not insights.get('product_insights') and len(all_ins) > 3:
            insights['product_insights'] = all_ins[3:6]
    except Exception:
        st.error('Failed to parse insights.')
        insights = {'category_insights':[], 'product_insights':[], 'insights':[]}
        insights = {'category_insights':[], 'product_insights':[], 'insights':[]}

        # Category chart & insights
    st.subheader("Category Performance")
    cat_chart = alt.Chart(category_summary).mark_bar().encode(
        x=alt.X('total_sales:Q', title='Total Sales'),
        y=alt.Y('Category:N', sort='-x', title=None)
    ).properties(height=300)
    st.altair_chart(cat_chart, use_container_width=True)

    st.markdown("### Category Insights")
    for entry in insights.get('category_insights', [])[:3]:
        if isinstance(entry, dict):
            text = entry.get('insight') or entry.get('text') or json.dumps(entry)
        else:
            text = entry
        st.markdown(f"- {text}")
    st.markdown("---")

    # Product chart & insights
    st.subheader("Top & Bottom SKU Movers")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Top Movers**")
        top_chart = alt.Chart(top_df).mark_bar().encode(
            x=alt.X('sales:Q', title='Sales'),
            y=alt.Y(f'{item_col}:N', sort='-x', title=None)
        ).properties(height=300)
        st.altair_chart(top_chart, use_container_width=True)
    with p2:
        st.markdown("**Cold Movers**")
        bot_chart = alt.Chart(bottom_df).mark_bar().encode(
            x=alt.X('sales:Q', title='Sales'),
            y=alt.Y(f'{item_col}:N', sort='x', title=None)
        ).properties(height=300)
        st.altair_chart(bot_chart, use_container_width=True)

    st.markdown("### Product Insights")
    for entry in insights.get('product_insights', [])[:3]:
        if isinstance(entry, dict):
            text = entry.get('insight') or entry.get('text') or json.dumps(entry)
        else:
            text = entry
        st.markdown(f"- {text}")
    st.markdown("---")

    # Final AI insights
    st.markdown("### AI Forecasts & Strategy Nudges")
    for line in insights.get('insights', [])[:3]:
        st.markdown(f"- {line}")

