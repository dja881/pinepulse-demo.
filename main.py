# --- REFINED AI PROMPT SECTION FOR PinePulse ---
# Example JSON schema to illustrate the exact output format and recommended actions
schema_example = {
    "category_insights": [
        "Tell me which category is accelerating or decelerating, why, and a 1–2 sentence action (e.g. 'run a 10% off promo').",
        "Identify the category with the highest days_supply and suggest an inventory tactic.",
        "Highlight the top-performing category and recommend a cross-sell or bundle opportunity."
    ],
    "product_insights": [
        "Identify one SKU at risk of stock-out and suggest reorder timing based on velocity and days_supply.",
        "Identify one SKU with excess days_supply and recommend a promotional tactic to clear stock.",
        "Identify one emerging fast-mover and suggest a bundling or upsell opportunity."
    ],
    "insights": [
        "Recommend one pricing adjustment based on payment-method trends (e.g., wallet, card).",
        "Recommend one marketing channel or discount strategy to boost performance of cold-movers.",
        "Recommend one inventory optimization tactic to reduce holding costs or improve turnover."
    ]
}

# Refined prompt string
prompt = f"""
You are a data-driven retail analyst. Output ONLY valid JSON matching exactly these keys:
  • category_insights: 3 bullet-point strings
  • product_insights: 3 bullet-point strings
  • insights: 3 bullet-point strings

Each bullet must:
  - Reference actual numbers from the data (sales, velocity, days_supply)
  - Include a one-sentence, actionable recommendation

Schema example:
{json.dumps(schema_example, indent=2)}

Category summary (name, total_sales, percent_of_total):
{json.dumps(category_summary.to_dict('records'), indent=2)}

Top SKUs (name, sales, quantity, velocity, days_supply):
{json.dumps(top_ctx, indent=2)}

Cold SKUs (name, sales, quantity, velocity, days_supply):
{json.dumps(bot_ctx, indent=2)}
"""

# When calling the AI:
resp = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=[
        {'role': 'system', 'content': 'Output only JSON.'},
        {'role': 'user', 'content': prompt}
    ],
    temperature=0.2,
    max_tokens=1000
)

