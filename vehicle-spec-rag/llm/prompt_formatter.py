
def prompt_formatter_gemini(query: str, context_items: list[dict]) -> str:
    """
    Formats the query and context into a prompt for the Gemini model.
    
    Args:
        query: User query string.
        context_items: List of dicts containing 'sentence_chunk'.
        
    Returns:
        Formatted prompt string.
    """
    # Convert context items to bullet list
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    prompt = f"""
You are an expert automotive service manual assistant.
You extract structured specifications from noisy context.

Follow these rules:
- Use ONLY the given context.
- Think silently, but output ONLY the final answer JSON.
- Always follow the example answer format exactly.
- If multiple matching components exist, output multiple JSON objects.

Below are examples of the expected answer style:

Example 1:
Query: Torque for brake caliper bolts
Answer: {{
    "component": "Brake Caliper Bolt",
    "spec_type": "Torque",
    "value": "35",
    "unit": "Nm"
}}

Example 2:
Query: Torque for brake disc shield bolts
Answer: {{
    "component": "Brake Disc Shield Bolt",
    "spec_type": "Torque",
    "value": "17",
    "unit": "Nm"
}}

Example 3:
Query: Torque for lower arm forward and rearward nuts
Answer: {{
    "component": "Lower Arm Forward and Rearward Nuts",
    "spec_type": "Torque",
    "value": "350",
    "unit": "Nm"
}}

Example 4:
Query: Torque for lower ball joint nut
Answer: {{
    "component": "Lower Ball Joint Nut",
    "spec_type": "Torque",
    "value": "175",
    "unit": "Nm"
}}

Example 5:
Query: Torque for shock absorber lower nuts
Answer: {{
    "component": "Shock Absorber Lower Nuts",
    "spec_type": "Torque",
    "value": "90",
    "unit": "Nm"
}}

Example 6:
Query: Torque for shock absorber upper mount nuts
Answer: {{
    "component": "Shock Absorber Upper Mount Nuts",
    "spec_type": "Torque",
    "value": "63",
    "unit": "Nm"
}}

Example 7:
Query: Torque for tie-rod end nut
Answer: {{
    "component": "Tie-Rod End Nut",
    "spec_type": "Torque",
    "value": "115",
    "unit": "Nm"
}}

Example 8:
Query: Torque for stabilizer bar bracket nuts
Answer: {{
    "component": "Stabilizer Bar Bracket Nuts",
    "spec_type": "Torque",
    "value": "55",
    "unit": "Nm"
}}

Example 9:
Query: Torque for stabilizer bar link nuts
Answer: {{
    "component": "Stabilizer Bar Link Nuts",
    "spec_type": "Torque",
    "value": "70",
    "unit": "Nm"
}}

Example 10:
Query: Torque for wheel speed sensor bolt
Answer: {{
    "component": "Wheel Speed Sensor Bolt",
    "spec_type": "Torque",
    "value": "18",
    "unit": "Nm"
}}

---------------------------------------------
Now use the following context items to answer the user query:

{context}

---------------------------------------------
User Query:
{query}

Return ONLY JSON:
"""

    return prompt
