import os
import streamlit as st
import pandas as pd
import openai

# --- INITIALIZE AI CLIENT ---
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- APP CONFIG ---
st.set_page_config(page_title="PinePulse - Weekly Store Pulse", layout="wide")
st.title("Weekly Store Pulse")

# --- DATA PATHS & LOADING ---
DATA_DIR = os.path.join(os.getcwd(), "data")
csv_paths = {
    "Kirana": os.path.join(DATA_DIR, "Kirana_Store_Transactions_v2.csv"),
    "Chemist": os.path.join(DATA_DIR, "Chemist_Store_Transactions_v2.csv"),
    "Cafe": os.path.join(DATA_DIR, "Cafe_Store_Transactions_v2.csv"),
    "Clothes": os.path.join(DATA_DIR, "Clothes_Store_Transactions_v2.csv"),
}

@st.cache_data
def load_data(paths):
    data = {}
    for name, path in paths.items():
        if os.path.isfile(path):
            data[name] = pd.read_csv(path, parse_dates=["Timestamp"])
    return data

all_data = load_data(csv_paths)

# --- USER INTERFACE ---
store_type = st.selectbox("Select Store Category", list(all_data.keys()))
if store_type:
    df = all_data[store_type]
    location = df["Location"].dropna().iloc[0] if "Location" in df.columns else ""
    if st.button("Generate Store Pulse"):
        prompt = f"""
Weekly Store Pulse: {store_type} Store — {location}
(Analyzed from recent 20 days of transaction data)

Hot-Selling SKUs (Restock Urgently)
- Provide top 3 SKUs by sales velocity and recommended restock actions.

Cold Movers (Consider Discount/Bundling)
- List slowest 3 SKUs with bundling or discount suggestions.

Footfall Patterns & Opportunity Slots
- Highlight slowest days and hours, with promo ideas.

External Signals & Trends
- Note relevant external trends affecting inventory.

Projected Next-Month Sales Forecast
- Forecast % change for key categories.

AI Nudges to Action This Week
- Summarize top 5 actionable items.
"""
        with st.spinner("Generating report…"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You are a concise retail analyst. Provide plain headings and bullet points, no emojis."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.6,
                max_tokens=500
            )
        report_md = response.choices[0].message.content.strip()
        # Render markdown directly for a clean, professional look
        st.markdown(report_md)

