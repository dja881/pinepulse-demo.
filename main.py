import os
import math
import streamlit as st
import pandas as pd
import openai
import altair as alt
import json
import re

# --- INITIALIZE AI CLIENT ---
client = openai.OpenAI(api_key=st.secrets['openai']['api_key'])

# --- APP CONFIG ---
st.set_page_config(page_title='PinePulse Dashboard', layout='wide')
st.title('📊 PinePulse - Weekly Store Pulse')

# --- DATA LOADING ---
DATA_DIR = os.path.join(os.getcwd(), 'data')
csv_paths = {
    'Kirana': os.path.join(DATA_DIR, 'Kirana_Store_Transactions_v2.csv'),
    'Chemist': os.path.join(DATA_DIR, 'Chemist_Store_Transactions_v2.csv'),
    'Cafe': os.path.join(DATA_DIR, 'Cafe_Store_Transactions_v2.csv'),
    'Clothes': os.path.join(DATA_DIR, 'Clothes_Store_Transactions_v2.csv')
}

@st.cache_data
def load_data():
    data = {}
    for name, path in csv_paths.items():
        if os.path.isfile(path):
            data[name] = pd.read_csv(path, parse_dates=['Timestamp'])
    return data

all_data = load_data()

# --- SIDEBAR ---
st.sidebar.header('Configuration')
source = st.sidebar.radio('Data Source:', ['Demo Data', 'Upload CSV'])
if source == 'Upload CSV':
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df_all = pd.read_csv(uploaded, parse_dates=['Timestamp'])
    else:
        st.stop()
else:
    store_type = st.sidebar.selectbox('Demo Store', list(all_data.keys()))
    df_all = all_data[store_type]

# --- TIME FILTER ---
days = st.sidebar.selectbox('Past days to include:', [7, 14, 30], index=0)
cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
df_all = df_all[df_all['Timestamp'] >= cutoff]

# --- COLUMN DETECTION ---
def find_col(keywords, cols):
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None

amount_col = find_col(['total amount', 'amount', 'total'], df_all.columns)
qty_col = find_col(['stock remaining', 'quantity'], df_all.columns)
item_col = find_col(['product name', 'sku'], df_all.columns)
cat_col = 'Category'

# --- DATA PREVIEW ---
st.markdown('### Data Preview')
st.dataframe(df_all.head(10))

# --- GENERATE REPORT ---
if st.sidebar.button('Generate Report'):
    df = df_all.copy()
    # Metrics
    total_sales = df[amount_col].sum()
    trans_count = len(df)
    unique_items = df[item_col].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Sales', f'₹{total_sales:,.0f}')
    c2.metric('Transactions', trans_count)
    c3.metric('Unique Products', unique_items)
    st.markdown('---')

    # Summaries
    sku_sales = df.groupby(item_col).agg(sales=(amount_col, 'sum')).reset_index()
    top_n = max(1, math.ceil(len(sku_sales) * 0.3))
    top_df = sku_sales.nlargest(top_n, 'sales')
    bottom_df = sku_sales.nsmallest(top_n, 'sales')
    category_summary = df.groupby(cat_col).agg(total_sales=(amount_col, 'sum')).reset_index()

    # Inventory context
    if qty_col:
        inv = df.groupby(item_col)[qty_col].sum().reset_index().rename(columns={qty_col: 'quantity'})
    else:
        inv = pd.DataFrame({item_col: top_df[item_col], 'quantity': [None] * len(top_df)})

    def build_ctx(sub_df):
        ctx = sub_df.merge(inv, on=item_col, how='left')
        ctx['velocity'] = (ctx['sales'] / days).round(1)
        ctx['days_supply'] = ctx.apply(
            lambda r: round(r['quantity'] / r['velocity'], 1) if r['quantity'] and r['velocity'] else None,
            axis=1
        )
        return ctx.to_dict('records')

    top_ctx = build_ctx(top_df)
    bot_ctx = build_ctx(bottom_df)

    # --- REFINED AI PROMPT (NO INTERNAL FIELD NAMES) ---
    schema_example = {
        "category_top_insights": [
            "Identify a high-growth category, explain the trend using actual sales numbers and average daily sales, and recommend a marketing tactic.",
            "Identify a slowing category, explain the decline with sales figures and stock duration, and recommend an action.",
            "Recommend a bundle or cross-sell opportunity for the leading category using recent performance data."
        ],
        "category_bottom_insights": [
            "Point out a category with excess stock based on current sales pace, and suggest a clearance strategy.",
            "Highlight a low-performing category with sales figures, and recommend a targeted discount or campaign.",
            "Suggest one channel or promotion to boost lagging category performance using recent metrics."
        ],
        "product_top_insights": [
            "Pick a top SKU nearing stock-out, describe remaining days of stock in plain English, and recommend reorder timing.",
            "Identify a best-selling SKU, reference its average daily units sold, and suggest a bundling option.",
            "Suggest a price tweak for a high-turnover SKU, referencing payment method trends."
        ],
        "product_bottom_insights": [
            "Pick a slow-moving SKU with high inventory days, describe stock duration simply, and recommend a promotion.",
            "Highlight a cold SKU by its recent sales figure, and suggest a targeted marketing channel to clear it.",
            "Recommend an inventory tactic like reducing reorder frequency for a low-sales SKU."
        ],
        "insights": [
            "Recommend a pricing adjustment based on payment-method share.",
            "Recommend a marketing channel or discount strategy for low-performing items.",
            "Recommend an inventory optimization tactic to improve turnover or reduce costs."
        ]
    }

    prompt = f"""
You are a data-driven retail analyst. Output ONLY valid JSON matching these keys:
  • category_top_insights: 3 bullet strings
  • category_bottom_insights: 3 bullet strings
  • product_top_insights: 3 bullet strings
  • product_bottom_insights: 3 bullet strings
  • insights: 3 bullet strings

Each bullet must:
  - Use plain English for metrics (e.g., 'average daily sales of 50 units', 'stock will last 5 days')
  - Reference actual numbers from the data
  - Include a one-sentence, actionable recommendation

Avoid using internal field names like 'velocity' or 'days_supply'.

Schema example:
{json.dumps(schema_example, indent=2)}

Category summary:
{json.dumps(category_summary.to_dict('records'), indent=2)}

Top SKUs:
{json.dumps(top_ctx, indent=2)}

Cold SKUs:
{json.dumps(bot_ctx, indent=2)}
"""

    resp = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {'role': 'system', 'content': 'Output only JSON.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.2,
        max_tokens=1200
    )

    raw = resp.choices[0].message.content
    st.text_area('Raw AI output', raw, height=200)

    match = re.search(r"\{[\s\S]*\}", raw)
    json_str = match.group(0) if match else raw
    try:
        data = json.loads(json_str)
    except Exception as e:
        st.error(f'Failed to parse insights: {e}')
        data = {
            'category_top_insights': [],
            'category_bottom_insights': [],
            'product_top_insights': [],
            'product_bottom_insights': [],
            'insights': []
        }

    # 1. Category Performance
    st.header('Category Performance')
    st.subheader('Top Category Insights')
    for line in data.get('category_top_insights', []): st.markdown(f'- {line}')
    st.subheader('Bottom Category Insights')
    for line in data.get('category_bottom_insights', []): st.markdown(f'- {line}')
    st.markdown('---')

    # 2. Product Movement
    st.header('Top & Bottom SKU Movers')
    p1, p2 = st.columns(2)
    with p1:
        st.subheader('Top SKU Insights')
        for line in data.get('product_top_insights', []): st.markdown(f'- {line}')
    with p2:
        st.subheader('Bottom SKU Insights')
        for line in data.get('product_bottom_insights', []): st.markdown(f'- {line}')
    st.markdown('---')

    # 3. AI Forecasts & Strategy Nudges
    st.subheader('AI Forecasts & Strategy Nudges')
    for line in data.get('insights', []): st.markdown(f'- {line}')
