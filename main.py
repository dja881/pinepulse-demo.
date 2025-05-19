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
    'Kirana':  os.path.join(DATA_DIR, 'Kirana_Store_Transactions_v2.csv'),
    'Cafe':    os.path.join(DATA_DIR, 'Cafe_Store_Transactions_v2.csv'),
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

# --- SIDEBAR & FILTERS ---
st.sidebar.header('Configuration')
source = st.sidebar.radio('Choose Data Source:', ['Demo Data', 'Upload CSV'])
if source == 'Upload CSV':
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df_all = pd.read_csv(uploaded, parse_dates=['Timestamp'])
    else:
        st.stop()
else:
    store_type = st.sidebar.selectbox('Demo Store', list(all_data.keys()))
    df_all = all_data[store_type]

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
qty_col    = find_col(['stock remaining', 'quantity'], df_all.columns)
item_col   = find_col(['product name', 'sku'], df_all.columns)
cat_col    = 'Category'

# --- DATA PREVIEW ---
st.markdown('### Data Preview')
st.dataframe(df_all.head(10))

# --- MAIN REPORT ---
if st.sidebar.button('Generate Report'):
    df = df_all.copy()

    # Metrics
    total_sales  = df[amount_col].sum()
    trans_count  = len(df)
    unique_items = df[item_col].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Sales',    f'₹{total_sales:,.0f}')
    c2.metric('Transactions',    trans_count)
    c3.metric('Unique Products', unique_items)
    st.markdown('---')

    # Summaries
    sku_sales = df.groupby(item_col).agg(sales=(amount_col, 'sum')).reset_index()
    top_n     = max(1, math.ceil(len(sku_sales) * 0.3))
    top_df    = sku_sales.nlargest(top_n, 'sales')
    bottom_df = sku_sales.nsmallest(top_n, 'sales')
    category_summary = df.groupby(cat_col).agg(total_sales=(amount_col, 'sum')).reset_index()

    # Inventory context
    if qty_col:
        inv = (df.groupby(item_col)[qty_col]
                 .sum()
                 .reset_index()
                 .rename(columns={qty_col: 'quantity'}))
    else:
        inv = pd.DataFrame({item_col: top_df[item_col], 'quantity': [None] * len(top_df)})

    def build_ctx(sub_df):
        ctx = sub_df.merge(inv, on=item_col, how='left')
        ctx['velocity']     = (ctx['sales'] / days).round(1)
        ctx['days_supply']  = ctx.apply(
            lambda r: round(r['quantity'] / r['velocity'], 1)
                      if r['quantity'] and r['velocity'] else None,
            axis=1
        )
        return ctx.to_dict('records')

    top_ctx = build_ctx(top_df)
    bot_ctx = build_ctx(bottom_df)

    # Prompt schema & AI call
    schema = {
        'category_top_insights':    ['…3 templates for top categories…'],
        'category_bottom_insights': ['…3 templates for bottom categories…'],
        'product_top_insights':     ['…3 templates for top SKUs…'],
        'product_bottom_insights':  ['…3 templates for bottom SKUs…'],
        'strategy_nudges':          ['…5 analytical, season/festival-aware templates…']
    }

    prompt = f"""
You are a data-driven retail analyst. Output ONLY JSON with these keys:
  • category_top_insights    (3 bullets)
  • category_bottom_insights (3 bullets)
  • product_top_insights     (3 bullets)
  • product_bottom_insights  (3 bullets)
  • strategy_nudges          (5 bullets, trend/festival-aware)

Each bullet should:
  – reference real numbers (sales, stock left, etc.) in plain English
  – include a one-sentence, actionable recommendation

Schema example:
{json.dumps(schema, indent=2)}

Category summary:
{json.dumps(category_summary.to_dict('records'), indent=2)}

Top SKUs context:
{json.dumps(top_ctx, indent=2)}

Cold SKUs context:
{json.dumps(bot_ctx, indent=2)}
"""

    resp = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {'role': 'system', 'content': 'Output only JSON.'},
            {'role': 'user',   'content': prompt}
        ],
        temperature=0.2,
        max_tokens=1200
    )
    raw = resp.choices[0].message.content
    match = re.search(r"\{[\s\S]*\}", raw)
    data  = json.loads(match.group(0)) if match else {}

    # 1. Category performance
    st.header('Category Performance')
    cat_chart = (alt.Chart(category_summary)
                   .mark_bar()
                   .encode(
                       x=alt.X('total_sales:Q', title='Sales'),
                       y=alt.Y(f'{cat_col}:N', sort='-x')
                   )
                   .properties(height=300))
    st.altair_chart(cat_chart, use_container_width=True)

    st.subheader('Top Category Insights')
    for line in data.get('category_top_insights', []):
        st.markdown(f'- {line}')
    st.subheader('Bottom Category Insights')
    for line in data.get('category_bottom_insights', []):
        st.markdown(f'- {line}')
    st.markdown('---')

    # 2. SKU charts and insights
    st.header('Top & Bottom SKU Movers')
    p1, p2 = st.columns(2)
    with p1:
        st.subheader('Top Movers')
        top_chart = (alt.Chart(top_df)
                       .mark_bar()
                       .encode(
                           x='sales:Q',
                           y=alt.Y(f'{item_col}:N', sort='-x')
                       )
                       .properties(height=300))
        st.altair_chart(top_chart, use_container_width=True)
        st.subheader('Top SKU Insights')
        for line in data.get('product_top_insights', []):
            st.markdown(f'- {line}')
    with p2:
        st.subheader('Cold Movers')
        cold_chart = (alt.Chart(bottom_df)
                        .mark_bar()
                        .encode(
                            x='sales:Q',
                            y=alt.Y(f'{item_col}:N', sort='x')
                        )
                        .properties(height=300))
        st.altair_chart(cold_chart, use_container_width=True)
        st.subheader('Cold SKU Insights')
        for line in data.get('product_bottom_insights', []):
            st.markdown(f'- {line}')
    st.markdown('---')

    # 3. Strategy nudges
    st.header('AI Forecasts & Strategy Nudges')
    for line in data.get('strategy_nudges', []):
        st.markdown(f'- {line}')
