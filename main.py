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
    pinecone.create_index(index_name, dimension=1536)
index = pinecone.Index(index_name)

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

    # --- PINECONE UPSERT CONTEXT ---
    embedding_model = 'text-embedding-ada-002'
    vectors = []
    for rec in top_ctx + bot_ctx:
        vec_resp = client.embeddings.create(model=embedding_model, input=json.dumps(rec))
        vector = vec_resp['data'][0]['embedding']
        vectors.append((rec[item_col], vector, rec))
    index.upsert(vectors=vectors)

    # --- OPTIONAL: RETRIEVE RELATED CONTEXT ---
    # query_text = 'Retrieve relevant store context'
    # q_resp = client.embeddings.create(model=embedding_model, input=[query_text])
    # q_vec = q_resp['data'][0]['embedding']
    # results = index.query(vector=q_vec, top_k=5, include_metadata=True)
    # retrieved = [match['metadata'] for match in results['matches']]

    # --- REFINED AI PROMPT (WITH PINECONE CONTEXT) ---
    schema_example = {
        "category_top_insights": [
            "Identify a high-growth category, explain the trend using actual sales and daily averages, and recommend a marketing tactic.",
            "Identify a slowing category, describe its sales drop, and suggest an immediate action.",
            "Recommend a bundle or cross-sell for the leading category based on recent performance."
        ],
        "category_bottom_insights": [
            "Point out a category with excess inventory relative to sales pace, and propose a clearance strategy.",
            "Highlight a low-performing category with its sales figure, and recommend a targeted discount or campaign.",
            "Suggest one channel or promotion to boost lagging category performance."
        ],
        "product_top_insights": [
            "Pick a top SKU nearing stock-out, describe remaining stock in plain English, and recommend reorder timing.",
            "Identify a best-selling SKU with its average daily sold units, and suggest a bundling option.

