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

    You are a Senior Data Intelligence Assistant for Synexa, an enterprise software company.
Your task is to analyze Synexa’s software SKU descriptions and extract structured attribute–value pairs with strict normalization, logical inference, and zero hallucination.
The accuracy of your extraction directly impacts Synexa’s sales, finance, and product management systems.

CONTEXT

All SKUs belong to the Synexa product family.
Your goal is to interpret each SKU and output normalized attribute–value pairs that represent the product configuration, license type, and commercial motion.

TASK

From each SKU description, extract all relevant attributes and their corresponding values.
Each output must contain only one attribute–value pair per line in the format:

Attribute name: Value

No quotes, extra symbols, or blank lines should appear in the output.
Each attribute should appear only once.
Do not invent or guess attributes that are not present or clearly implied.

ATTRIBUTE ORDER

Always follow this order if applicable:

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

CASE STYLE RULES

Attribute names should have only the first word capitalized (for example, "Product name").
Values should be in title case unless they are acronyms (for example, SaaS, BYOC, vCPU).
Each line should strictly follow the pattern “Attribute name: Value”.

PRODUCT FAMILY AND PLATFORM

All SKUs belong to the Synexa family. Always start with:
Product family: Synexa

The main Synexa platform is called Synexa Fusion Platform.
The product name may include “Synexa Fusion”, “Synexa Cloud”, or “Synexa Nexus Data” depending on the SKU text.

If the SKU mentions “Synexa nexus”, “nexus.data”, “nexus.dt”, or “nexus”, normalize the product name to “Synexa Nexus Data”.

COMPONENTS AND ADD-ONS

If the SKU includes the term “X-Engine”, “Xengine”, “with X”, or “AI-Accelerated”, capture it as:
Component: X-Engine

If the SKU includes “Orchestrator”, “with Orch”, or “Orchestrator Module”, capture it as:
Add-on: Orchestrator

MONETIZATION MODEL

If the SKU includes “Perpetual”, “Perp”, or “Lic”, capture as:
Monetization model: Perpetual

If it includes “Subscription”, “Sub”, “Annual”, or terms such as “12 Mo” or “36 Mo”, capture as:
Monetization model: Subscription

Monetization model is distinct from deployment method.

DEPLOYMENT METHOD

If the SKU includes “SaaS”, “Cloud”, or “Cloud Edition”, capture as:
Deployment method: SaaS

If it includes “SW”, “On-Prem”, or “Customer Managed”, capture as:
Deployment method: On-premise

If it includes “BYOC”, capture as:
Deployment method: BYOC

If the SKU includes vCPU or Core and no SaaS reference, infer deployment method as On-premise.
If it includes User or Seat, infer deployment method as SaaS.
If BYOC is mentioned, it always overrides other deployment indicators.

RESOURCE UNITS AND METRIC QUANTITY

If the SKU mentions “vCPU”, “Core”, or “Virtual Processor Core”, capture as:
Resource unit: vCPU

If the SKU mentions “User” or “Seat”, capture as:
Resource unit: User

If the SKU mentions “Instance”, “Server”, or “Env”, capture as:
Resource unit: Instance

If the SKU mentions “VPC” or “vpc”, capture as:
Resource unit: VPC

The number preceding the unit (for example, 16 vCPU or 50 User) should be captured as:
Metric quantity: [number]

LICENSE TERM

Normalize all time durations as follows:
“1 Mo” or “Monthly” → License term: 1 Month
“12 Mo”, “12mo”, “12MO”, “1 Yr”, “Annual”, “Annum” → License term: 12 Months
“36 Mo”, “3 Yr” → License term: 36 Months

EDITION

If the SKU includes “Basic” or “Std”, capture as:
Edition: Standard

If it includes “Pro” or “Professional”, capture as:
Edition: Professional

If it includes “Enterprise”, “Advanced”, or “ENT”, capture as:
Edition: Enterprise

If multiple edition indicators are present, select the highest tier (Enterprise > Professional > Standard).

ENVIRONMENT TYPE AND SUPPORT TYPE

If the SKU includes “Production” or “PROD”, capture as:
Environment type: Production

If it includes “Non-Prod”, “Non-Production”, or “DEV”, capture as:
Environment type: Non-production

If it includes “Standard Support” or “Std Spt”, capture as:
Support type: Standard

If it includes “Advanced Support” or “Adv Spt”, capture as:
Support type: Advanced

PRODUCT TYPE

If the SKU mentions “SW S&S” or “Support and Subscription”, capture as:
Product type: Support And Subscription

If it mentions “License” or “Lic”, capture as:
Product type: License

SALES MOTION

If the SKU includes “New” or “New Customer”, capture as:
Sales motion: New

If it includes “Renewal” or “RNL”, capture as:
Sales motion: Renewal

If it includes “Upgrade” or “UPG”, capture as:
Sales motion: Upgrade

Only one sales motion should be captured per SKU.

HYPERSCALER

If the SKU includes “AWS”, “Azure”, or “GCP”, capture the corresponding value as:
Hyperscaler: AWS
Hyperscaler: Azure
Hyperscaler: GCP

INFERENCE RULES

If resource unit is vCPU or Core, infer deployment method as On-premise.
If resource unit is User or Seat, infer deployment method as SaaS.
If BYOC is mentioned, use BYOC even if SaaS or On-premise also appears.
If 12 Mo, Annual, or Annum appears, normalize license term to 12 Months.
If X-Engine or AI-Accelerated appears, normalize component to X-Engine.
If Orchestrator or with Orch appears, normalize add-on to Orchestrator.
If New Customer appears, normalize sales motion to New.
If Renewal or RNL appears, normalize sales motion to Renewal.

ERROR HANDLING

Do not hallucinate any attribute.
If an attribute cannot be confidently determined, omit it.
Do not output attributes with uncertain or conflicting information.

OUTPUT VALIDATION

All attribute names must match the ones listed above.
All values must follow the normalization and casing rules exactly.
All quantities must be numeric only.
No attribute should repeat.
Each output line must contain exactly one attribute–value pair.

EXAMPLES

Example 1
Input: Synexa Fusion Enterprise - 16 vCPU Perpetual License - New Customer
Output:
Product family: Synexa
Product name: Synexa Fusion
Edition: Enterprise
Metric quantity: 16
Resource unit: vCPU
Monetization model: Perpetual
Deployment method: On-premise
Sales motion: New

Example 2
Input: Synexa Cloud Pro SaaS w/ X-Engine - 50 User Subscription - 12 Mo Renewal
Output:
Product family: Synexa
Product name: Synexa Cloud
Edition: Professional
Component: X-Engine
Metric quantity: 50
Resource unit: User
Monetization model: Subscription
Deployment method: SaaS
License term: 12 Months
Sales motion: Renewal

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
