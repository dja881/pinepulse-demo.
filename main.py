import os
import streamlit as st
import pandas as pd
import openai
import re

# --- INITIALIZE AI CLIENT ---
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- APP CONFIG ---
st.set_page_config(page_title="PinePulse - Weekly Store Pulse", layout="wide")
st.title("Weekly Store Pulse")

# --- DATA PATHS & LOADING ---
DATA_DIR = os.path.join(os.getcwd(), "data")
csv_paths = {
    "Kirana":   os.path.join(DATA_DIR, "Kirana_Store_Transactions_v2.csv"),
    "Chemist":  os.path.join(DATA_DIR, "Chemist_Store_Transactions_v2.csv"),
    "Cafe":     os.path.join(DATA_DIR, "Cafe_Store_Transactions_v2.csv"),
    "Clothes":  os.path.join(DATA_DIR, "Clothes_Store_Transactions_v2.csv"),
}

@st.cache_data
def load_data(paths):
    data = {}
    for name, path in paths.items():
        if os.path.isfile(path):
            data[name] = pd.read_csv(path, parse_dates=["Timestamp"], infer_datetime_format=True)
    return data

all_data = load_data(csv_paths)

# --- USER INTERFACE ---
store_type = st.selectbox("Select Store Category", list(all_data.keys()))
if store_type:
    df = all_data[store_type]
    location = df.get('Location', pd.Series()).dropna().iloc[0] if 'Location' in df.columns else ''

    if st.button("Generate Store Pulse"):
        # Build AI prompt without emojis for clarity
        pulse_prompt = f"""
Weekly Store Pulse: {store_type} Store — {location}
(Analyzed from recent 20 days of transaction data)

Hot-Selling SKUs (Restock Urgently)
[List top SKUs by velocity with bullet points and actionable restock guidance]

Cold Movers (Consider Discount/Bundling)
[List slowest moving SKUs with bullet points and promotion suggestions]

Footfall Patterns & Opportunity Slots
[Highlight slowest days/hours and suggested promos]

External Signals & Trends
[Call out external trends relevant to inventory or weather]

Projected Next-Month Sales Forecast
[Give % forecasts per category]

AI Nudges to Action This Week
[Checklist of top 3-5 actionable nudges]
"""
        with st.spinner("Generating AI-driven pulse report..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data-driven retail analyst."},
                    {"role": "user", "content": pulse_prompt}
                ],
                temperature=0.7,
                max_tokens=600,
            )
        raw = response.choices[0].message.content.strip()
        # parse sections by header lines
        pattern = r"^([A-Za-z].+[A-Za-z])\n((?:-[^\n]*\n?)+)"  # header then bullet lines
        sections = re.findall(pattern, raw, flags=re.MULTILINE)
        # layout in two columns for a minimal UI
        col1, col2 = st.columns(2)
        for idx, (header, body) in enumerate(sections):
            target = col1 if idx < len(sections)/2 else col2
            with target:
                st.subheader(header)
                for line in body.strip().split("\n"):
                    text = line.lstrip('- ').strip()
                    st.markdown(f"- {text}")

