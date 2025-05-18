sku_prompt = f'''
You are a retail analyst tasked with understanding why certain products are performing better or worse than others.
Your goal is to reason through the differences using data (velocity, stock, sales) and potential trends such as:
- payment preferences (e.g. UPI vs card)
- regional differences
- time patterns
- product appeal or bundling
- seasonal or festival impact

Top Products context:
{json.dumps(top_context, indent=2)}

Slow Products context:
{json.dumps(bottom_context, indent=2)}

Payment Summary:
{json.dumps(payment_summary.to_dict(orient="records"), indent=2)}

Give 3 clear and specific recommendations for each top and bottom product.
Explain why each product might be performing the way it is — using logical reasoning and correlations.
If applicable, identify any seasonality, regional or pricing trends.

Respond only with valid JSON in this format:
{
  "top_recos": [
    {"sku": "Product Name", "recommendations": ["rec 1", "rec 2", "rec 3"]}
  ],
  "bottom_recos": [
    {"sku": "Product Name", "recommendations": ["rec 1", "rec 2", "rec 3"]}
  ],
  "insights": ["trend insight 1", "insight 2", "insight 3", "insight 4"],
  "product_insights": ["product insight 1", "product insight 2"],
  "payment_insights": ["payment behavior 1", "payment behavior 2"]
}
'''
