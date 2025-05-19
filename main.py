import os
import math
import streamlit as st
import pandas as pd
import openai
import pinecone
import altair as alt
import json
import re

# --- INITIALIZE CLIENTS ---
openai_api_key = st.secrets['openai']['api_key']
client = openai.OpenAI(api_key=openai_api_key)

pinecone_api_key = st.secrets['pinecone']['api_key']
pinecone_env = st.secrets['pinecone']['environment']
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index_name = 'pinepulse-context'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536)
index = pinecone.Index(index_name)

# --- APP CONFIG ---
st.set_page_config(page_title='📊 PinePulse - Weekly Store Pulse', layout='wide')
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
    bottom_ctx = build_ctx(bottom_df)

    # --- PINECONE UPSERT CONTEXT ---
    embedding_model = 'text-embedding-ada-002'
    vectors = []
    for rec in top_ctx + bottom_ctx:
        resp_embed = client.embeddings.create(model=embedding_model, input=json.dumps(rec))
        vector = resp_embed['data'][0]['embedding']
        vectors.append((rec[item_col], vector, rec))
    index.upsert(vectors=vectors)

    # --- REFINED AI PROMPT ---
    schema_example = {
        "category_top_insights": [
            "Identify a high-growth category, explain the trend with actual sales and average daily sales, and recommend an action.",
            "Spot a category with slowing momentum, describe its decline in plain English, and suggest an immediate tactic.",
            "Recommend one cross-sell or bundle opportunity for the leading category based on recent performance."
        ],
        "category_bottom_insights": [
            "Highlight a low-performing category with its sales figure, and propose a clearance strategy.",
            "Call out a category with excess stock relative to its sales pace, and recommend a discount or campaign.",
            "Suggest one marketing channel to boost the lagging category."
        ],
        "product_top_insights": [
            "Select a top SKU nearing stock-out, describe remaining stock simply, and recommend reorder timing.",
            "Identify a best-selling SKU with its average daily sold units, and suggest a bundle or upsell.",
            "Recommend a pricing tweak for a fast-moving SKU based on payment method trends."
        ],
        "product_bottom_insights": [
            "Identify a slow-moving SKU with surplus stock days, describe in plain English, and recommend a promo.",
            "Highlight a cold SKU by its recent sales count, and suggest a targeted marketing channel.",
            "Recommend an inventory adjustment like changing reorder frequency for a low-sales SKU."
        ],
        "insights": [
            "Forecast a temperature-driven surge in Cold Drinks next month and recommend increasing stock by 30%.",
            "Leverage the upcoming festival season by bundling Ethnic Wear with Accessories, anticipating a 25% demand uptick.",
            "Expect monsoon to dampen Footwear sales; initiate a weather-themed promotion to maintain growth.",
            "Note a 15% jump in digital wallet payments; launch a wallet-exclusive flash sale for high-margin items.",
            "Prepare for peak summer by boosting Ice Cream inventory 40% ahead of average and running a seasonal offer."
        ]
    }

    prompt = f"""
You are a data-driven retail analyst. Output ONLY valid JSON matching these keys:
  • category_top_insights: 3 bullet strings
  • category_bottom_insights: 3 bullet strings
  • product_top_insights: 3 bullet strings
  • product_bottom_insights: 3 bullet strings
  • insights: 5 bullet strings

Each bullet must:
  - Use plain English for metrics (e.g., 'average daily sales of 50 units', 'stock will last five days')
  - Reference actual numbers from the data
  - Include a one-sentence, actionable recommendation

Schema example:
{json.dumps(schema_example, indent=2)}

Category summary:
{json.dumps(category_summary.to_dict('records'), indent=2)}

Top SKUs:
{json.dumps(top_ctx, indent=2)}

Cold SKUs:
{json.dumps(bottom_ctx, indent=2)}
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

    # Parse AI response
    raw = resp.choices[0].message.content
    match = re.search(r"\{[\s\S]*\}", raw)
    json_str = match.group(0) if match else raw
    try:
        data = json.loads(json_str)
    except Exception as e:
        st.error(f'Failed to parse insights: {e}')
        data = {key: [] for key in schema_example.keys()}

    # 1. Category Performance
    st.header('Category Performance')
    cat_chart = alt.Chart(category_summary).mark_bar().encode(
        x='total_sales:Q', y=alt.Y(f"{cat_col}:N", sort='-x')
    ).properties(height=300)
    st.altair_chart(cat_chart, use_container_width=True)

    st.subheader('Top Category Insights')
    for line in data.get('category_top_insights', []):
        st.markdown(f'- {line}')

    st.subheader('Bottom Category Insights')
    for line in data.get('category_bottom_insights', []):
        st.markdown(f'- {line}')

    st.markdown('---')

    # 2. SKU Performance
    st.header('SKU Movers')
    lhs, rhs = st.columns(2)
    with lhs:
        st.subheader('Top SKUs')
        top_chart = alt.Chart(top_df).mark_bar().encode(
            x='sales:Q', y=alt.Y(f"{item_col}:N", sort='-x')
        ).properties(height=300)
        st.altair_chart(top_chart, use_container_width=True)
        st.subheader('Top SKU Insights')
        for line in data.get('product_top_insights', []):
            st.markdown(f'- {line}')
    with rhs:
        st.subheader('Cold SKUs')
        cold_chart = alt.Chart(bottom_df).mark_bar().encode(
            x='sales:Q', y=alt.Y(f"{item_col}:N", sort='x')
        ).properties(height=300)
        st.altair_chart(cold_chart, use_container_width=True)
        st.subheader('Bottom SKU Insights')
        for line in data.get('product_bottom_insights', []):
            st.markdown(f'- {line}')

    st.markdown('---')

    # 3. AI Forecasts & Strategy Nudges
    st.header('AI Forecasts & Strategy Nudges')
    for line in data.get('insights', []):
        st.markdown(f'- {line}')
