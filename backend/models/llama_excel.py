import os
import pandas as pd
import ollama
import re
from collections import defaultdict
from difflib import get_close_matches


model_name = "llama3"


def process_excel_with_llama(filepath, domain_prompt=None):
    import difflib

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Input file '{filepath}' not found!")

    df = pd.read_excel(filepath)
    if "SKU_Description" not in df.columns:
        raise KeyError("Excel must contain a column named 'SKU_Description'")

    # ✅ Keep your strong domain context
    domain_context = (
        f"\nDomain Context:\n{domain_prompt}\n"
        if domain_prompt
        else "\nYou are working in a general product data context.\n"
    )

    # ---------------------
    # 💡 Your high-accuracy LLaMA prompt builder
    # ---------------------
    def extract_attributes(description):
        prompt = f"""{domain_prompt}

   
ROLE

You are Synexa’s Senior Data Intelligence Assistant.
You extract strict, normalized attribute–value pairs from Synexa SKU descriptions with zero hallucination.
Your output populates sales, finance, and product systems.
Every extracted attribute must be correct, normalized, and fully justified by the text.


TASK

Given a SKU description, extract all clearly present or strongly implied attributes.
Output one attribute–value pair per line in the EXACT format:
Attribute name: Value



No quotes, no extra text, no blank lines.
If an attribute cannot be determined with certainty, omit it.

GLOBAL RULES
1. Product family
Always output:
Product family: Synexa

2. Product name
Normalize as:
Contains "Synexa Fusion" → Synexa Fusion
Contains "Synexa Cloud" or "Cloud Edition" → Synexa Cloud
Contains "Nexus", "Nexus.Data", "Nexus.DT", "Synexa Nexus" → Synexa Nexus Data

3. Edition
Normalize:
Basic, Std → Standard
Pro, Professional → Professional
Enterprise, Advanced, ENT → Enterprise
If multiple appear, choose highest tier (Enterprise > Professional > Standard).

4. Component / Add-on

Contains “X-Engine”, “Xengine”, “with X”, “AI-Accelerated” →
Component: X-Engine

Contains “Orchestrator”, “with Orch”, “Orchestrator Module” →
Add-on: Orchestrator

5. Metric quantity + Resource unit

Detect the numeric quantity and unit:

Normalize units:

vCPU, Core, Virtual Processor Core → vCPU

User, Seat → User

Instance, Server, Env → Instance

VPC, vpc → VPC

Output:

Metric quantity: <number>
Resource unit: <unit>

6. Monetization model

Perpetual, Perp, Lic → Perpetual

Subscription, Sub, Annual, 12 Mo, 36 Mo → Subscription

7. Deployment method

Normalize as:

SaaS, Cloud → SaaS

SW, On-Prem, Customer Managed → On-premise

BYOC → BYOC (always overrides)

Inference rules:

If unit = vCPU/Core → On-premise

If unit = User/Seat → SaaS

BYOC always overrides

8. License term

Normalize:

1 Mo → 1 Month

“12 Mo”, “12mo”, “1 Yr”, “Annual”, “Annum” → 12 Months

“36 Mo”, “3 Yr” → 36 Months

9. Product type

“SW S&S”, “Support and Subscription” → Support And Subscription

“License”, “Lic” → License

10. Environment type

Production, PROD → Production

Non-Prod, Non-Production, DEV → Non-production

11. Support type

Standard Support, Std Spt → Standard

Advanced Support, Adv Spt → Advanced

12. Hyperscaler

AWS, Azure, GCP → normalize as-is

13. Sales motion

Select one:

New, New Customer → New

Renewal, RNL → Renewal

Upgrade, UPG → Upgrade

OUTPUT ORDER (strict)

Always output in this exact order when applicable:

Product family

Product name

Edition

Component

Add-on

Metric quantity

Resource unit

Monetization model

Deployment method

License term

Product type

Environment type

Support type

Hyperscaler

Sales motion

OUTPUT RULES



No repeated attributes.

No attributes without evidence.

Title Case values (except acronyms like SaaS, BYOC, vCPU).

Exact attribute names (do not create new ones).

Each line = one attribute–value pair.

FINAL INSTRUCTION

Analyze the given SKU description.
Apply all normalization and inference rules strictly.
Return only valid, normalized attribute–value pairs, one per line, in the correct order.
Do not add any commentary, explanation, or formatting other than the attribute–value pairs.

        SKU Description:
        {description}


    """

        try:
            response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    def parse_attributes(text):
        attr_dict = defaultdict(list)
        for line in text.splitlines():
            match = re.match(r"(.+?)\s*[:=]\s*(.+)", line.strip())
            if match:
                attr, val = match.groups()
                attr_dict[attr.strip()].append(val.strip())
        return attr_dict

    def normalize_attr_name(attr):
        attr = re.sub(r"[^a-zA-Z0-9 ]", "", attr)
        attr = re.sub(r"\s+", " ", attr).strip()
        attr = attr.title()
        return attr

    global_attributes = defaultdict(set)
    total = len(df)
    print(f"📊 Starting extraction for {total} SKUs using LLaMA...", flush=True)

    for i, row in df.iterrows():
        desc = str(row["SKU_Description"]).strip()
        if not desc:
            continue

        print(f"\n🔍 Processing row {i+1}/{total}: {desc}", flush=True)
        raw_output = extract_attributes(desc)
        attr_dict = parse_attributes(raw_output)

        for attr, vals in attr_dict.items():
            normalized_attr = normalize_attr_name(attr)
            for v in vals:
                global_attributes[normalized_attr].add(v.strip())

    merged_attributes = defaultdict(set)
    all_attrs = list(global_attributes.keys())

    for attr in all_attrs:
        found_match = False
        for canonical in list(merged_attributes.keys()):
            if difflib.SequenceMatcher(None, attr.lower(), canonical.lower()).ratio() > 0.85:
                merged_attributes[canonical].update(global_attributes[attr])
                found_match = True
                break

        if not found_match:
            merged_attributes[attr].update(global_attributes[attr])

    rows = []
    max_values = max((len(v) for v in merged_attributes.values()), default=0)
    for attr, vals in merged_attributes.items():
        rows.append([attr] + list(vals) + [""] * (max_values - len(vals)))

    columns = ["Attribute"] + [f"Value{i+1}" for i in range(max_values)]

    print(f"✅ Extraction finished. Total unique merged attributes: {len(rows)}", flush=True)
    return {"columns": columns, "rows": rows}


# ⚡ NEW FUNCTION — Row-by-row extraction using same strong prompt
def process_excel_row_with_llama(sku_text, domain_prompt=None):
    """
    Process a single SKU description using LLaMA.
    Returns a list of extracted [Attribute, Value] pairs.
    """
    if not sku_text or sku_text.strip() == "":
        return []

    # ✅ Use the exact same prompt that worked well for you
    prompt = f"""{domain_prompt}

    SKU Description:
    {sku_text}

    Return only the attribute-value lines, nothing else.
    """

    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        text = response["message"]["content"].strip()

        # 🧩 Parse the output into pairs
        rows = []
        for line in text.splitlines():
            match = re.match(r"(.+?)\s*[:=]\s*(.+)", line.strip())
            if match:
                attr, val = match.groups()
                rows.append([attr.strip(), val.strip()])

        return rows

    except Exception as e:
        print(f"❌ Row processing error: {e}")
        return []


def refine_with_llama(selected_rows, chat_history, full_table):
    import ollama
    import json

    prompt = (
        f"You are refining a table of attributes and values. "
        f"Here are the rows the user selected: {selected_rows}. "
        f"The user instruction is: {chat_history[-1]['content']}. "
        f"Return ONLY the corrected attribute name and value pairs "
        f"in plain JSON array format, like this:\n"
        f'[["Attribute", "Value"], ["Another Attribute", "Value"]]'
    )

    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    output_text = response["message"]["content"]

    try:
        refined_rows = json.loads(output_text)
    except json.JSONDecodeError:
        lines = [line for line in output_text.split("\n") if "=" in line]
        refined_rows = []
        for line in lines:
            parts = line.split("=")
            if len(parts) >= 2:
                refined_rows.append([parts[0].strip(), "=".join(parts[1:]).strip()])

    for i, row in enumerate(selected_rows):
        if row in full_table:
            idx = full_table.index(row)
            if i < len(refined_rows):
                full_table[idx] = refined_rows[i]

    return full_table
