import os
import streamlit as st
import pandas as pd
import openai
import altair as alt

# --- INITIALIZE AI CLIENT ---
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- APP CONFIG ---
st.set_page_config(page_title="PinePulse - Interactive Store Dashboard", layout="wide")
st.title("📊 PinePulse - Weekly Store Pulse Dashboard")

# --- DATA PATHS & LOADING ---
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
store_name = None

if store_type:
    df_all = all_data[store_type]
    stores = sorted(df_all["Store Name"].unique().tolist())
    store_name = st.sidebar.selectbox("Store Name", stores)
    if st.sidebar.button("Generate Report"):
        df = df_all[df_all["Store Name"] == store_name]

        # --- BENCHMARK METRICS ---
        total_sales = df["Txn Amount (₹)"].sum()
        num_txn = len(df)
        unique_products = df["Product Name"].nunique()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sales", f"₹{total_sales:,.0f}")
        m2.metric("Transactions", num_txn)
        m3.metric("Unique Products", unique_products)

        st.markdown("---")

        # --- TOP & BOTTOM SKUs ---
        sku_sales = df.groupby("Product Name")["Txn Amount (₹)"].sum().reset_index()
        top5 = sku_sales.sort_values(by="Txn Amount (₹)", ascending=False).head(5)
        bot5 = sku_sales.sort_values(by="Txn Amount (₹)", ascending=True).head(5)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Hot-Selling SKUs")
            chart_top = alt.Chart(top5).mark_bar().encode(
                x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                y=alt.Y("Product Name:N", sort="-x", title=None)
            ).properties(height=300)
            st.altair_chart(chart_top, use_container_width=True)
        with c2:
            st.subheader("Cold Movers")
            chart_bot = alt.Chart(bot5).mark_bar(color="#FFA500").encode(
                x=alt.X("Txn Amount (₹):Q", title="Sales (₹)"),
                y=alt.Y("Product Name:N", sort="x", title=None)
            ).properties(height=300)
            st.altair_chart(chart_bot, use_container_width=True)

        st.markdown("---")

        # --- FOOTFALL PATTERNS ---
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
            day_chart = alt.Chart(by_day).mark_line(point=True).encode(
                x="Weekday:N", y="count:Q"
            ).properties(height=250)
            st.altair_chart(day_chart, use_container_width=True)
        with c4:
            st.subheader("Transactions by Hour")
            hour_chart = alt.Chart(by_hour).mark_line(point=True).encode(
                x="Hour:O", y="count:Q"
            ).properties(height=250)
            st.altair_chart(hour_chart, use_container_width=True)

        st.markdown("---")

        # --- CATEGORY MIX PIE ---
        cat_mix = df.groupby("Product Category")["Txn Amount (₹)"].sum().reset_index()
        pie = alt.Chart(cat_mix).mark_arc().encode(
            theta="Txn Amount (₹):Q", color="Product Category:N"
        ).properties(height=300)
        st.subheader("Category Revenue Mix")
        st.altair_chart(pie, use_container_width=True)

        st.markdown("---")

        # --- AI INSIGHT ---
        insight_prompt = f"Provide a concise one-sentence insight about {store_name} based on {num_txn} transactions of the past 20 days."
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are a data-driven business analyst."},
                {"role":"user","content":insight_prompt}
            ],
            temperature=0.7,
            max_tokens=60
        )
        insight = resp.choices[0].message.content.strip()
        st.subheader("AI Insight")
        st.write(insight)
