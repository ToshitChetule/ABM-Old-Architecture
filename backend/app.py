


# from flask import Flask, request, jsonify, send_file
# import os
# from flask_cors import CORS
# import pandas as pd
# from variant_analysis import run_variant_analysis
# import json
# from datetime import datetime

# # 🧠 Imports
# from models.llama_excel import process_excel_row_with_llama
# from models.mistral_pdf import process_pdf_with_mistral_normalizer
# from graph.neo4j_builder import Neo4jBuilder
# from models.refine_graph import refine_with_graph_context

# # Rule engine (new)
# from models.rule_engine import generate_rules_from_sku_matrix, save_rules_to_json

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# UPLOAD_DIR = "uploads"
# OUTPUT_DIR = "outputs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # In-memory caches for last processed data
# LAST_SKU_MATRIX = None
# LAST_RULESET = None


# # 🧠 Domain Prompt Function (same as before)
# def get_domain_prompt(industry, product_type):
#     base_prompt = f"""

# ROLE

# You are Synexa’s Senior Data Intelligence Assistant.
# You extract strict, normalized attribute–value pairs from Synexa SKU descriptions with zero hallucination.
# Your output populates sales, finance, and product systems.
# Every extracted attribute must be correct, normalized, and fully justified by the text.


# TASK

# Given a SKU description, extract all clearly present or strongly implied attributes.
# Output one attribute–value pair per line in the EXACT format:
# Attribute name: Value



# No quotes, no extra text, no blank lines.
# If an attribute cannot be determined with certainty, omit it.

# GLOBAL RULES
# 1. Product family
# Always output:
# Product family: Synexa

# 2. Product name
# Normalize as:
# Contains "Synexa Fusion" → Synexa Fusion
# Contains "Synexa Cloud" or "Cloud Edition" → Synexa Cloud
# Contains "Nexus", "Nexus.Data", "Nexus.DT", "Synexa Nexus" → Synexa Nexus Data

# 3. Edition
# Normalize:
# Basic, Std → Standard
# Pro, Professional → Professional
# Enterprise, Advanced, ENT → Enterprise
# If multiple appear, choose highest tier (Enterprise > Professional > Standard).

# 4. Component / Add-on

# Contains “X-Engine”, “Xengine”, “with X”, “AI-Accelerated” →
# Component: X-Engine

# Contains “Orchestrator”, “with Orch”, “Orchestrator Module” →
# Add-on: Orchestrator

# 5. Metric quantity + Resource unit

# Detect the numeric quantity and unit:

# Normalize units:

# vCPU, Core, Virtual Processor Core → vCPU

# User, Seat → User

# Instance, Server, Env → Instance

# VPC, vpc → VPC

# Output:

# Metric quantity: <number>
# Resource unit: <unit>

# 6. Monetization model

# Perpetual, Perp, Lic → Perpetual

# Subscription, Sub, Annual, 12 Mo, 36 Mo → Subscription

# 7. Deployment method

# Normalize as:

# SaaS, Cloud → SaaS

# SW, On-Prem, Customer Managed → On-premise

# BYOC → BYOC (always overrides)

# Inference rules:

# If unit = vCPU/Core → On-premise

# If unit = User/Seat → SaaS

# BYOC always overrides

# 8. License term

# Normalize:

# 1 Mo → 1 Month

# “12 Mo”, “12mo”, “1 Yr”, “Annual”, “Annum” → 12 Months

# “36 Mo”, “3 Yr” → 36 Months

# 9. Product type

# “SW S&S”, “Support and Subscription” → Support And Subscription

# “License”, “Lic” → License

# 10. Environment type

# Production, PROD → Production

# Non-Prod, Non-Production, DEV → Non-production

# 11. Support type

# Standard Support, Std Spt → Standard

# Advanced Support, Adv Spt → Advanced

# 12. Hyperscaler

# AWS, Azure, GCP → normalize as-is

# 13. Sales motion

# Select one:

# New, New Customer → New

# Renewal, RNL → Renewal

# Upgrade, UPG → Upgrade

# OUTPUT ORDER (strict)

# Always output in this exact order when applicable:

# Product family

# Product name

# Edition

# Component

# Add-on

# Metric quantity

# Resource unit

# Monetization model

# Deployment method

# License term

# Product type

# Environment type

# Support type

# Hyperscaler

# Sales motion

# OUTPUT RULES



# No repeated attributes.

# No attributes without evidence.

# Title Case values (except acronyms like SaaS, BYOC, vCPU).

# Exact attribute names (do not create new ones).

# Each line = one attribute–value pair.

#     ### CONTEXT
#     Industry: {industry if industry else "general"}
#     Product Type: {product_type if product_type else "unspecified"}
#     """

#     domain_prompts = {
#         "automotive": """
#         Focus on vehicle specifications:
#         - Make, Model, Trim, Year
#         - Engine details, Power (HP), Torque (Nm), Transmission
#         - Fuel type, Tank capacity, Mileage
#         - Dimensions, Weight, Ground clearance
#         - Compatibility and Part number
#         """,
#         "pharmaceuticals": """
#         Focus on:
#         - Brand and Generic Name
#         - Strength (mg/ml)
#         - Dosage Form (Tablet, Capsule, Syrup)
#         - Ingredients, Packaging type, Quantity
#         - Manufacturer, Expiry date, Batch/Lot, Therapeutic category
#         """,
#         "electronics": """
#         Focus on:
#         - Brand, Model number, Series
#         - Power, Voltage, Frequency, Capacity (GB/TB)
#         - Battery, Display size, Resolution
#         - Connectivity (Wi-Fi, Bluetooth, HDMI)
#         - Warranty, Material, Weight, Dimensions
#         """,
#         "food_beverages": """
#         Focus on:
#         - Product name, Brand
#         - Ingredients, Nutritional values
#         - Net weight/volume, Flavor
#         - Packaging type/material, Shelf life
#         - Manufacturer, Country of origin
#         """,
#         "chemical": """
#         Focus on:
#         - Chemical name, Formula, Purity, CAS number
#         - Physical form, Molecular weight, Boiling/Melting point
#         - Applications, Packaging, Safety classification
#         """
#     }

#     if industry and industry.lower() in domain_prompts:
#         base_prompt += domain_prompts[industry.lower()]
#     else:
#         base_prompt += "\nExtract all relevant descriptive and technical attributes clearly."

#     return base_prompt


# # 🧾 Main File Processing Endpoint
# @app.route("/process", methods=["POST"])
# def process_file():
#     """Handles Excel or PDF upload and returns both SKU-level and aggregated data."""
#     global LAST_SKU_MATRIX, LAST_RULESET

#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     filename = file.filename
#     if not filename:
#         return jsonify({"error": "Invalid filename"}), 400

#     # Extract user context
#     industry = request.form.get("industry", "general")
#     product_type = request.form.get("productType", "")
#     domain_prompt = get_domain_prompt(industry, product_type)

#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     ext = os.path.splitext(filename)[1].lower()
#     print(f"\n📂 Received file: {filename} ({ext})")

#     try:
#         # ✅ Excel Processing
#         if ext in [".xlsx", ".xls"]:
#             df = pd.read_excel(filepath)

#             if "SKU_Description" not in df.columns:
#                 return jsonify({"error": "Missing 'SKU_Description' column in Excel."}), 400

#             total_rows = len(df)
#             print(f"🚀 Processing {total_rows} SKU rows using LLaMA...\n")

#             attribute_map = {}
#             sku_matrix = []  # 🆕 For SKU-level matrix

#             for i, row in df.iterrows():
#                 sku_text = str(row["SKU_Description"]).strip()
#                 if not sku_text:
#                     continue

#                 print(f"🧠 Processing row {i + 1}/{total_rows}: {sku_text[:100]}...")
#                 extracted_pairs = process_excel_row_with_llama(sku_text, domain_prompt)

#                 # 🧩 Store SKU-level data
#                 sku_matrix.append({
#                     "sku": sku_text,
#                     "attributes": extracted_pairs
#                 })

#                 # 🧩 Build aggregated attribute map
#                 for attr, val in extracted_pairs:
#                     if attr not in attribute_map:
#                         attribute_map[attr] = set()
#                     for v in str(val).split(","):
#                         clean_val = v.strip()
#                         if clean_val:
#                             attribute_map[attr].add(clean_val)

#             # 🧮 Build Aggregated Matrix
#             max_values = max(len(vals) for vals in attribute_map.values()) if attribute_map else 0
#             columns = ["Attribute"] + [f"Value{i + 1}" for i in range(max_values)]

#             rows = []
#             for attr, vals in attribute_map.items():
#                 val_list = list(vals)
#                 val_list += [""] * (max_values - len(val_list))
#                 rows.append([attr] + val_list)

#             print("\n✅ Row-by-row Excel processing complete with multi-value support.\n")

#             # 🕸️ Push extracted attributes to Neo4j Knowledge Graph (optional)
#            # 🕸️ Push extracted attributes to Neo4j Knowledge Graph (optional)
#             try:
#                 print("📡 Connecting to Neo4j for Knowledge Graph update...")
#                 neo = Neo4jBuilder()

#                  # 🧹 Clear any previous data before inserting the new session
#                 neo.clear_database()
#                 print("🧹 Cleared old data from Neo4j (session-only mode).")
#                 neo.add_attribute_value_pairs(attribute_map)
#                 neo.close()
#                 print("✅ Knowledge Graph successfully updated.\n")
#             except Exception as graph_error:
#                 print(f"⚠️ Neo4j update skipped due to error: {graph_error}")

#             # --- Generate configuration rules from sku_matrix (backend)
#             try:
#                 print("🧾 Generating configuration rules from SKU matrix...")
#                 LAST_SKU_MATRIX = sku_matrix
#                 rule_res = generate_rules_from_sku_matrix(
#                     sku_matrix,
#                     min_confidence=0.85,
#                     min_row_count=3,
#                     min_support_pct=0.0,
#                     max_examples=5,
#                     persist_json=False
#                 )
#                 LAST_RULESET = rule_res
#                 print(f"✅ Generated {len(rule_res.get('rules', []))} rules.")
#             except Exception as re_err:
#                 print(f"⚠️ Rule generation failed: {re_err}")
#                 LAST_RULESET = {"rules": [], "generated_at": datetime.utcnow().isoformat() + "Z"}

#             # ✅ Return both data types to frontend
#             return jsonify({
#                 "sku_matrix": sku_matrix,  # Per-SKU extracted data
#                 "aggregated_matrix": {     # Unique attribute-value pairs
#                     "columns": columns,
#                     "rows": rows
#                 },
#                 "rules": LAST_RULESET,
#                 "model_used": "llama3",
#                 "industry": industry,
#                 "product_type": product_type
#             })


#         elif ext == ".pdf":
#             print("🚀 Running Mistral + Normalizer extraction...\n")
#             result = process_pdf_with_mistral_normalizer(filepath, domain_prompt)

#             return jsonify({
#                 "sku_matrix": [],
#                 "aggregated_matrix": {
#                     "columns": result.get("columns", []),
#                     "rows": result.get("rows", [])
#                 },
#                 "model_used": "Mistral + Normalizer",
#                 "industry": industry,
#                 "product_type": product_type
#             })



#         # else:
#         #     return jsonify({"error": f"Unsupported file type: {ext}"}), 400

#     except Exception as e:
#         print(f"❌ Error during processing: {str(e)}")
#         return jsonify({"error": str(e)}), 500


# def detect_refinement_intent(user_prompt: str):
#     if not user_prompt:
#         return "unknown"
#     p = user_prompt.lower()
#     if "rename" in p and "attribute" in p:
#         return "attribute"
#     if "rename" in p and "value" in p:
#         return "value"
#     if "under" in p or "in " in p or "inside" in p:
#         return "value"
#     return "attribute"

# @app.route("/refine_graph", methods=["POST", "OPTIONS"])
# def refine_graph():
#     if request.method == "OPTIONS":
#         return jsonify({"status": "OK"}), 200

#     try:
#         from graph.neo4j_builder import Neo4jBuilder
#         import re

#         data = request.get_json()
#         attributes = data.get("attributes", [])
#         if isinstance(attributes, str):
#             attributes = [attributes]

#         prompt = data.get("prompt", "").strip()
#         if not prompt:
#             return jsonify({"error": "Prompt missing"}), 400

#         neo = Neo4jBuilder()

#         # --- Step 1: Get existing attributes ---
#         with neo.driver.session() as session:
#             res = session.run("MATCH (a:Attribute) RETURN a.name AS name")
#             all_attrs = [r["name"] for r in res]

#         print("📘 Attributes in graph:", all_attrs)
#         print("📥 Selected in UI:", attributes)
#         print("🧠 Prompt:", prompt)

#         # --- Step 2: Split into clear atomic commands ---
#         # Split prompt into clauses safely at 'and', 'then', ',' or '.'
#         raw_actions = re.split(r"\s*(?:and|then|,|\.)\s*", prompt, flags=re.IGNORECASE)
#         raw_actions = [a.strip() for a in raw_actions if a.strip()]
#         print(f"🧩 Split atomic actions: {raw_actions}")

#         performed_actions = []

#         # --- Step 3: Process each atomic action sequentially ---
#         for act in raw_actions:
#             act_low = act.lower()

#             # ✅ RENAME ATTRIBUTE
#             if act_low.startswith("rename attribute"):
#                 m = re.match(
#                     r"rename\s+attribute\s+([\w\s\-]+?)\s+to\s+([\w\s\-]+)",
#                     act, re.IGNORECASE
#                 )
#                 if m:
#                     old_attr, new_attr = m.groups()
#                     neo.rename_attribute(old_attr.strip(), new_attr.strip())
#                     performed_actions.append(f"Renamed attribute '{old_attr}' → '{new_attr}'")

#             # ✅ RENAME VALUE
#             elif act_low.startswith("rename value"):
#                 m = re.match(
#                     r"rename\s+value\s+([\w\s\-]+?)\s+to\s+([\w\s\-]+)",
#                     act, re.IGNORECASE
#                 )
#                 if m:
#                     old_val, new_val = m.groups()
#                     for attribute in attributes:
#                         values = neo.get_values(attribute)
#                         if any(v.lower() == old_val.lower() for v in values):
#                             neo.rename_value(attribute, old_val, new_val)
#                             performed_actions.append(f"Renamed value '{old_val}' → '{new_val}' under '{attribute}'")

#             # ✅ GENERIC RENAME (rename from X to Y)
#             elif act_low.startswith("rename from"):
#                 m = re.match(
#                     r"rename\s+from\s+([\w\s\-]+?)\s+to\s+([\w\s\-]+)",
#                     act, re.IGNORECASE
#                 )
#                 if m:
#                     old, new = m.groups()
#                     if old.lower() in [a.lower() for a in all_attrs]:
#                         neo.rename_attribute(old.strip(), new.strip())
#                         performed_actions.append(f"Renamed attribute '{old}' → '{new}' (generic)")
#                     else:
#                         for attribute in attributes:
#                             values = neo.get_values(attribute)
#                             if any(v.lower() == old.lower() for v in values):
#                                 neo.rename_value(attribute, old, new)
#                                 performed_actions.append(f"Renamed value '{old}' → '{new}' under '{attribute}' (generic)")

#             # ✅ ADD VALUE
#             elif act_low.startswith("add value"):
#                 m = re.match(
#                     r"add\s+value\s+([\w\s\-]+?)\s+under\s+([\w\s\-]+)",
#                     act, re.IGNORECASE
#                 )
#                 if m:
#                     val, attr = m.groups()
#                     neo.add_value(attr.strip(), val.strip())
#                     performed_actions.append(f"Added value '{val}' under '{attr}'")

#             # ✅ REMOVE VALUE
#             elif act_low.startswith("remove value"):
#                 m = re.match(
#                     r"remove\s+value\s+([\w\s\-]+?)\s+under\s+([\w\s\-]+)",
#                     act, re.IGNORECASE
#                 )
#                 if m:
#                     val, attr = m.groups()
#                     neo.remove_value(attr.strip(), val.strip())
#                     performed_actions.append(f"Removed value '{val}' under '{attr}'")

#             # ✅ DELETE ATTRIBUTE
#             elif act_low.startswith("delete attribute"):
#                 m = re.match(
#                     r"delete\s+attribute\s+([\w\s\-]+)",
#                     act, re.IGNORECASE
#                 )
#                 if m:
#                     attr = m.group(1)
#                     neo.delete_attribute(attr.strip())
#                     performed_actions.append(f"Deleted attribute '{attr}'")

#         # --- Step 4: Refresh Graph Data ---
#         with neo.driver.session() as session:
#             result = session.run("""
#                 MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
#                 RETURN a.name AS attribute, collect(v.value) AS values
#             """)
#             rows = [[r["attribute"]] + r["values"] for r in result]
#             max_len = max((len(r) - 1 for r in rows), default=0)
#             columns = ["Attribute"] + [f"Value{i+1}" for i in range(max_len)]

#         # After refine actions, recompute rules (because attributes/values changed)
#         try:
#             print("🔁 Recomputing configuration rules after refinement...")
#             if LAST_SKU_MATRIX:
#                 LAST_RULESET = generate_rules_from_sku_matrix(
#                     LAST_SKU_MATRIX,
#                     min_confidence=0.85,
#                     min_row_count=3,
#                     min_support_pct=0.0,
#                     max_examples=5,
#                     persist_json=False
#                 )
#                 print(f"✅ Recomputed {len(LAST_RULESET.get('rules', []))} rules.")
#             else:
#                 print("⚠️ No SKU matrix available to recompute rules.")
#         except Exception as rr:
#             print(f"⚠️ Rule recompute after refine failed: {rr}")

#         neo.close()

#         return jsonify({
#             "status": "success",
#             "actions": performed_actions,
#             "updated_context": {"columns": columns, "rows": rows},
#             "rules": LAST_RULESET
#         }), 200

#     except Exception as e:
#         print(f"❌ Error in refine_graph: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/process_variant", methods=["POST"])
# def process_variant():
#     try:
#         file = request.files.get("file")
#         if not file:
#             return jsonify({"success": False, "error": "No file uploaded."}), 400

#         input_path = os.path.join(UPLOAD_DIR, file.filename)
#         file.save(input_path)
#         output_path = run_variant_analysis(input_path, OUTPUT_DIR)

#         return jsonify({"success": True, "filename": os.path.basename(output_path)})

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500


# @app.route("/download/<filename>", methods=["GET"])
# def download_file(filename):
#     path = os.path.join(OUTPUT_DIR, filename)
#     if not os.path.exists(path):
#         return jsonify({"error": "File not found"}), 404
#     return send_file(path, as_attachment=True)


# @app.route("/compare/ui_vs_graph", methods=["POST"])
# def compare_ui_vs_graph():
#     try:
#         from graph.neo4j_builder import Neo4jBuilder
#         ui_attrs = set(request.json.get("attributes", []))
#         neo = Neo4jBuilder()

#         with neo.driver.session() as session:
#             res = session.run("MATCH (a:Attribute) RETURN a.name AS attr")
#             graph_attrs = {r["attr"] for r in res}

#         neo.close()

#         missing_in_graph = sorted(ui_attrs - graph_attrs)
#         extra_in_graph = sorted(graph_attrs - ui_attrs)

#         return jsonify({
#             "ui_count": len(ui_attrs),
#             "graph_count": len(graph_attrs),
#             "missing_in_graph": missing_in_graph,
#             "extra_in_graph": extra_in_graph
#         })

#     except Exception as e:
#         print(f"❌ Comparison error: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/graph/aggregated", methods=["GET"])
# def get_aggregated_from_graph():
#     try:
#         from graph.neo4j_builder import Neo4jBuilder
#         neo = Neo4jBuilder()

#         with neo.driver.session() as session:
#             result = session.run("""
#                 MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
#                 RETURN a.name AS attribute, collect(v.value) AS values
#             """)

#             rows = []
#             attribute_map = {}
#             for record in result:
#                 attribute = record["attribute"]
#                 values = record["values"]
#                 attribute_map[attribute] = values
#                 rows.append([attribute] + values)

#             max_len = max((len(v) for v in attribute_map.values()), default=0)
#             columns = ["Attribute"] + [f"Value{i+1}" for i in range(max_len)]

#         neo.close()
#         return jsonify({"columns": columns, "rows": rows}), 200

#     except Exception as e:
#         print(f"❌ Error fetching graph data: {e}")
#         return jsonify({"error": str(e)}), 500


# # ---------------------------
# # Rules API Endpoints (new)
# # ---------------------------

# @app.route("/rules/configuration", methods=["GET"])
# def get_configuration_rules():
#     global LAST_RULESET, LAST_SKU_MATRIX
#     try:
#         if not LAST_RULESET:
#             if LAST_SKU_MATRIX:
#                 LAST_RULESET = generate_rules_from_sku_matrix(LAST_SKU_MATRIX)
#             else:
#                 return jsonify({"rules": [], "message": "No SKU data available to generate rules."}), 200

#         return jsonify(LAST_RULESET), 200

#     except Exception as e:
#         print(f"❌ Error fetching rules: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/rules/recompute", methods=["POST"])
# def recompute_rules():
#     global LAST_RULESET, LAST_SKU_MATRIX
#     try:
#         payload = request.get_json(silent=True) or {}
#         sku_matrix = payload.get("sku_matrix") or LAST_SKU_MATRIX
#         if not sku_matrix:
#             return jsonify({"error": "No sku_matrix provided and no previous data available."}), 400

#         min_confidence = float(payload.get("min_confidence", 0.85))
#         min_row_count = int(payload.get("min_row_count", 3))
#         min_support_pct = float(payload.get("min_support_pct", 0.0))
#         max_examples = int(payload.get("max_examples", 5))
#         persist_json = bool(payload.get("persist_json", False))

#         rule_res = generate_rules_from_sku_matrix(
#             sku_matrix,
#             min_confidence=min_confidence,
#             min_row_count=min_row_count,
#             min_support_pct=min_support_pct,
#             max_examples=max_examples,
#             persist_json=persist_json
#         )
#         LAST_SKU_MATRIX = sku_matrix

#         LAST_RULESET = rule_res

#         return jsonify(rule_res), 200

#     except Exception as e:
#         print(f"❌ Error recomputing rules: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/rules/<rule_id>/preview", methods=["GET"])
# def preview_rule(rule_id):
#     global LAST_RULESET
#     try:
#         if not LAST_RULESET:
#             return jsonify({"error": "No rules available"}), 404
#         found = next((r for r in LAST_RULESET.get("rules", []) if r["id"] == rule_id), None)
#         if not found:
#             return jsonify({"error": "Rule not found"}), 404
#         return jsonify({"examples": found.get("examples", []), "rule": found}), 200
#     except Exception as e:
#         print(f"❌ Error previewing rule: {e}")
#         return jsonify({"error": str(e)}), 500


# @app.route("/graph/rebuild_sku_matrix", methods=["GET"])
# def rebuild_sku_matrix_from_graph():
#     """
#     Rebuilds SKU matrix from Neo4j:
#     ONE SKU → ALL ATTRIBUTES → ALL VALUES
#     """
#     try:
#         from graph.neo4j_builder import Neo4jBuilder
#         neo = Neo4jBuilder()

#         with neo.driver.session() as session:
#             result = session.run("""
#                 MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
#                 RETURN a.name AS attribute, collect(v.value) AS values
#             """)

#             sku_matrix = []

#             # Build one combined SKU row
#             attributes = []
#             for record in result:
#                 attributes.append([record["attribute"], ", ".join(record["values"])])

#             sku_matrix.append({
#                 "sku": "CLEANED-SKU",
#                 "attributes": attributes
#             })

#         neo.close()
#         return jsonify({ "sku_matrix": sku_matrix })

#     except Exception as e:
#         print("Error rebuilding sku matrix:", e)
#         return jsonify({"error": str(e)}), 500



# if __name__ == "__main__":
#     app.run(debug=True, port=5000)



















# app.py (Final Version: persists SKU nodes and rebuilds per-SKU matrix from Neo4j)
from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS
import pandas as pd
from variant_analysis import run_variant_analysis
import json
from datetime import datetime

# 🧠 Imports
from models.llama_excel import process_excel_row_with_llama
from models.mistral_pdf import process_pdf_with_mistral_normalizer
from graph.neo4j_builder import Neo4jBuilder
from models.refine_graph import refine_with_graph_context

# Rule engine (new)
from models.rule_engine import generate_rules_from_sku_matrix, save_rules_to_json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory caches for last processed data
LAST_SKU_MATRIX = None
LAST_RULESET = None



# 🧠 Domain Prompt Function (same as before)
def get_domain_prompt(industry, product_type):
    base_prompt = f"""
     # Instructions for the Model

    You are processing SKU descriptions at scale. Extract attributes and values that are clearly stated or strongly implied.
    Normalize abbreviations, shorthand, and codes into human-readable attributes and values.
    Merge duplicate attributes (textual variants or semantic duplicates) into a single attribute using a consistent Title Case name.
    Prioritize commercial, monetization, and marketing attributes when present.
    

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



    ### CONTEXT
    Industry: {industry if industry else "general"}
    Product Type: {product_type if product_type else "unspecified"}
    """

    domain_prompts = {
        "automotive": """
        Focus on vehicle specifications:
        - Make, Model, Trim, Year
        - Engine details, Power (HP), Torque (Nm), Transmission
        - Fuel type, Tank capacity, Mileage
        - Dimensions, Weight, Ground clearance
        - Compatibility and Part number
        """,
        "pharmaceuticals": """
        Focus on:
        - Brand and Generic Name
        - Strength (mg/ml)
        - Dosage Form (Tablet, Capsule, Syrup)
        - Ingredients, Packaging type, Quantity
        - Manufacturer, Expiry date, Batch/Lot, Therapeutic category
        """,
        "electronics": """
        Focus on:
        - Brand, Model number, Series
        - Power, Voltage, Frequency, Capacity (GB/TB)
        - Battery, Display size, Resolution
        - Connectivity (Wi-Fi, Bluetooth, HDMI)
        - Warranty, Material, Weight, Dimensions
        """,
        "food_beverages": """
        Focus on:
        - Product name, Brand
        - Ingredients, Nutritional values
        - Net weight/volume, Flavor
        - Packaging type/material, Shelf life
        - Manufacturer, Country of origin
        """,
        "chemical": """
        Focus on:
        - Chemical name, Formula, Purity, CAS number
        - Physical form, Molecular weight, Boiling/Melting point
        - Applications, Packaging, Safety classification
        """
    }

    if industry and industry.lower() in domain_prompts:
        base_prompt += domain_prompts[industry.lower()]
    else:
        base_prompt += "\nExtract all relevant descriptive and technical attributes clearly."

    return base_prompt


# -------------------------
# Helper: build sku_matrix from Neo4j (NEW)
# -------------------------
def load_sku_matrix_from_graph():
    """
    Query Neo4j for SKU nodes and reconstruct per-SKU attribute pairs:
      RETURN [{"sku": "text", "attributes": [[attr, val], ...]}, ...]
    Uses pattern:
      (s:SKU)-[:HAS_VALUE]->(v)<-[:HAS_VALUE]-(a:Attribute)
    """
    try:
        neo = Neo4jBuilder()
        sku_matrix = []
        with neo.driver.session() as session:
            result = session.run("""
                MATCH (s:SKU)
                OPTIONAL MATCH (s)-[:HAS_VALUE]->(v)<-[:HAS_VALUE]-(a:Attribute)
                WITH s, collect(CASE WHEN a IS NOT NULL AND v IS NOT NULL THEN [a.name, v.value] ELSE NULL END) AS pairs
                RETURN s.sku_text AS sku, [p IN pairs WHERE p IS NOT NULL | p] AS attributes
                ORDER BY s.sku_text
            """)
            for r in result:
                sku = r["sku"]
                attrs = r["attributes"] or []
                sku_matrix.append({"sku": sku, "attributes": attrs})
        neo.close()
        return sku_matrix
    except Exception as e:
        print(f"❌ Error loading sku matrix from graph: {e}")
        return []


# ---------------------------
# Main file processing
# ---------------------------
@app.route("/process", methods=["POST"])
def process_file():
    """Handles Excel or PDF upload and returns both SKU-level and aggregated data."""
    global LAST_SKU_MATRIX, LAST_RULESET

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    # Extract user context
    industry = request.form.get("industry", "general")
    product_type = request.form.get("productType", "")
    domain_prompt = get_domain_prompt(industry, product_type)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext = os.path.splitext(filename)[1].lower()
    print(f"\n📂 Received file: {filename} ({ext})")

    try:
        # Excel Processing
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)

            if "SKU_Description" not in df.columns:
                return jsonify({"error": "Missing 'SKU_Description' column in Excel."}), 400

            total_rows = len(df)
            print(f"🚀 Processing {total_rows} SKU rows using LLaMA...\n")

            attribute_map = {}
            sku_matrix = []  # For SKU-level matrix

            for i, row in df.iterrows():
                sku_text = str(row["SKU_Description"]).strip()
                if not sku_text:
                    continue

                print(f"🧠 Processing row {i + 1}/{total_rows}: {sku_text[:100]}...")
                extracted_pairs = process_excel_row_with_llama(sku_text, domain_prompt)

                # Store SKU-level data
                sku_matrix.append({
                    "sku": sku_text,
                    "attributes": extracted_pairs
                })

                # Build aggregated attribute map
                for attr, val in extracted_pairs:
                    if attr not in attribute_map:
                        attribute_map[attr] = set()
                    for v in str(val).split(","):
                        clean_val = v.strip()
                        if clean_val:
                            attribute_map[attr].add(clean_val)

            # Build aggregated matrix
            max_values = max(len(vals) for vals in attribute_map.values()) if attribute_map else 0
            columns = ["Attribute"] + [f"Value{i + 1}" for i in range(max_values)]

            rows = []
            for attr, vals in attribute_map.items():
                val_list = list(vals)
                val_list += [""] * (max_values - len(val_list))
                rows.append([attr] + val_list)

            print("\n✅ Row-by-row Excel processing complete with multi-value support.\n")

            # Push extracted attributes & SKUs to Neo4j (NEW behavior)
            try:
                print("📡 Connecting to Neo4j to persist SKU nodes & graph...")
                neo = Neo4jBuilder()

                # Clear previous session data if you want session replacement
                neo.clear_database()
                print("🧹 Cleared old data from Neo4j (session-only mode).")

                # Bulk insert SKUs -> Attributes -> Values and connect them
                with neo.driver.session() as session:
                    for s in sku_matrix:
                        sku_text = s["sku"]
                        # Create SKU node
                        session.run("MERGE (sku:SKU {sku_text: $sku_text})", sku_text=sku_text)

                        for attr, val in s["attributes"]:
                            if not attr or not val:
                                continue
                            val = str(val).strip()
                            attr_name = str(attr).strip()

                            # Support multi-values separated by commas (if any)
                            for single_val in [v.strip() for v in val.split(",") if v.strip()]:
                                session.run("""
                                    MERGE (a:Attribute {name: $attr})
                                    MERGE (v:Value {value: $val})
                                    MERGE (a)-[:HAS_VALUE]->(v)
                                    MERGE (sku:SKU {sku_text: $sku_text})
                                    MERGE (sku)-[:HAS_VALUE]->(v)
                                    MERGE (sku)-[:HAS_ATTRIBUTE]->(a)
                                """, attr=attr_name, val=single_val, sku_text=sku_text)
                neo.close()
                print("✅ Persisted SKU nodes + attributes to Neo4j.\n")
            except Exception as graph_error:
                print(f"⚠️ Neo4j update skipped/did not complete due to error: {graph_error}")

            # Generate configuration rules from sku_matrix (backend)
            try:
                print("🧾 Generating configuration rules from SKU matrix...")
                LAST_SKU_MATRIX = sku_matrix
                rule_res = generate_rules_from_sku_matrix(
                    sku_matrix,
                    min_confidence=0.85,
                    min_row_count=3,
                    min_support_pct=0.0,
                    max_examples=5,
                    persist_json=False
                )
                LAST_RULESET = rule_res
                print(f"✅ Generated {len(rule_res.get('rules', []))} rules.")
            except Exception as re_err:
                print(f"⚠️ Rule generation failed: {re_err}")
                LAST_RULESET = {"rules": [], "generated_at": datetime.utcnow().isoformat() + "Z"}

            # Return results
            return jsonify({
                "sku_matrix": sku_matrix,
                "aggregated_matrix": {
                    "columns": columns,
                    "rows": rows
                },
                "rules": LAST_RULESET,
                "model_used": "llama3",
                "industry": industry,
                "product_type": product_type
            })

        elif ext == ".pdf":
            print("🚀 Running Mistral + Normalizer extraction...\n")
            result = process_pdf_with_mistral_normalizer(filepath, domain_prompt)

            return jsonify({
                "sku_matrix": [],
                "aggregated_matrix": {
                    "columns": result.get("columns", []),
                    "rows": result.get("rows", [])
                },
                "model_used": "Mistral + Normalizer",
                "industry": industry,
                "product_type": product_type
            })

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Refinement endpoint
# ---------------------------
def detect_refinement_intent(user_prompt: str):
    if not user_prompt:
        return "unknown"
    p = user_prompt.lower()
    if "rename" in p and "attribute" in p:
        return "attribute"
    if "rename" in p and "value" in p:
        return "value"
    if "under" in p or "in " in p or "inside" in p:
        return "value"
    return "attribute"


@app.route("/refine_graph", methods=["POST", "OPTIONS"])
def refine_graph():
    if request.method == "OPTIONS":
        return jsonify({"status": "OK"}), 200

    global LAST_SKU_MATRIX, LAST_RULESET

    try:
        from graph.neo4j_builder import Neo4jBuilder
        import re

        data = request.get_json()
        attributes = data.get("attributes", [])
        if isinstance(attributes, str):
            attributes = [attributes]

        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt missing"}), 400

        neo = Neo4jBuilder()

        # Step 1: Fetch existing attributes in graph
        with neo.driver.session() as session:
            res = session.run("MATCH (a:Attribute) RETURN a.name AS name")
            all_attrs = [r["name"] for r in res]

        print("📘 Attributes in graph:", all_attrs)
        print("📥 Selected in UI:", attributes)
        print("🧠 Prompt:", prompt)

        # Split prompt into actions
        raw_actions = re.split(r"\s*(?:and|then|,|\.)\s*", prompt, flags=re.IGNORECASE)
        raw_actions = [a.strip() for a in raw_actions if a.strip()]
        print(f"🧩 Split atomic actions: {raw_actions}")

        performed_actions = []

        # Process actions (reuse your existing logic)
        for act in raw_actions:
            act_low = act.lower()

            # RENAME ATTRIBUTE
            if act_low.startswith("rename attribute"):
                m = re.match(r"rename\s+attribute\s+([\w\s\-]+?)\s+to\s+([\w\s\-]+)", act, re.IGNORECASE)
                if m:
                    old_attr, new_attr = m.groups()
                    neo.rename_attribute(old_attr.strip(), new_attr.strip())
                    performed_actions.append(f"Renamed attribute '{old_attr}' → '{new_attr}'")

            # RENAME VALUE
            elif act_low.startswith("rename value"):
                m = re.match(r"rename\s+value\s+([\w\s\-]+?)\s+to\s+([\w\s\-]+)", act, re.IGNORECASE)
                if m:
                    old_val, new_val = m.groups()
                    for attribute in attributes:
                        values = neo.get_values(attribute)
                        if any(v.lower() == old_val.lower() for v in values):
                            neo.rename_value(attribute, old_val, new_val)
                            performed_actions.append(f"Renamed value '{old_val}' → '{new_val}' under '{attribute}'")

            # GENERIC RENAME (rename from X to Y)
            elif act_low.startswith("rename from"):
                m = re.match(r"rename\s+from\s+([\w\s\-]+?)\s+to\s+([\w\s\-]+)", act, re.IGNORECASE)
                if m:
                    old, new = m.groups()
                    if old.lower() in [a.lower() for a in all_attrs]:
                        neo.rename_attribute(old.strip(), new.strip())
                        performed_actions.append(f"Renamed attribute '{old}' → '{new}' (generic)")
                    else:
                        for attribute in attributes:
                            values = neo.get_values(attribute)
                            if any(v.lower() == old.lower() for v in values):
                                neo.rename_value(attribute, old, new)
                                performed_actions.append(f"Renamed value '{old}' → '{new}' under '{attribute}' (generic)")

            # ADD VALUE
            elif act_low.startswith("add value"):
                m = re.match(r"add\s+value\s+([\w\s\-]+?)\s+under\s+([\w\s\-]+)", act, re.IGNORECASE)
                if m:
                    val, attr = m.groups()
                    neo.add_value(attr.strip(), val.strip())
                    performed_actions.append(f"Added value '{val}' under '{attr}'")

            # REMOVE VALUE
            elif act_low.startswith("remove value"):
                m = re.match(r"remove\s+value\s+([\w\s\-]+?)\s+under\s+([\w\s\-]+)", act, re.IGNORECASE)
                if m:
                    val, attr = m.groups()
                    neo.remove_value(attr.strip(), val.strip())
                    performed_actions.append(f"Removed value '{val}' under '{attr}'")

            # DELETE ATTRIBUTE
            elif act_low.startswith("delete attribute"):
                m = re.match(r"delete\s+attribute\s+([\w\s\-]+)", act, re.IGNORECASE)
                if m:
                    attr = m.group(1)
                    neo.delete_attribute(attr.strip())
                    performed_actions.append(f"Deleted attribute '{attr}'")

        # Refresh Graph Aggregated data for UI
        with neo.driver.session() as session:
            result = session.run("""
                MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
                RETURN a.name AS attribute, collect(v.value) AS values
            """)
            rows = [[r["attribute"]] + r["values"] for r in result]
            max_len = max((len(r) - 1 for r in rows), default=0)
            columns = ["Attribute"] + [f"Value{i+1}" for i in range(max_len)]

        # --- Step: rebuild LAST_SKU_MATRIX from graph so config matrix and rules reflect changes ---
        try:
            print("🔁 Rebuilding SKU matrix from Neo4j after refine...")
            LAST_SKU_MATRIX = load_sku_matrix_from_graph()
            print(f"✅ Loaded {len(LAST_SKU_MATRIX)} SKU rows from Neo4j.")
        except Exception as rr:
            print(f"⚠️ Failed to rebuild SKU matrix from graph: {rr}")

        # Recompute rules using updated SKU matrix
        try:
            print("🔁 Recomputing configuration rules after refinement...")
            if LAST_SKU_MATRIX:
                LAST_RULESET = generate_rules_from_sku_matrix(
                    LAST_SKU_MATRIX,
                    min_confidence=0.85,
                    min_row_count=3,
                    min_support_pct=0.0,
                    max_examples=5,
                    persist_json=False
                )
                print(f"✅ Recomputed {len(LAST_RULESET.get('rules', []))} rules.")
            else:
                print("⚠️ No SKU matrix available to recompute rules.")
        except Exception as rr:
            print(f"⚠️ Rule recompute after refine failed: {rr}")

        neo.close()

        return jsonify({
            "status": "success",
            "actions": performed_actions,
            "updated_context": {"columns": columns, "rows": rows},
            "rules": LAST_RULESET
        }), 200

    except Exception as e:
        print(f"❌ Error in refine_graph: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Variant / download endpoints unchanged
# ---------------------------
@app.route("/process_variant", methods=["POST"])
def process_variant():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"success": False, "error": "No file uploaded."}), 400

        input_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(input_path)
        output_path = run_variant_analysis(input_path, OUTPUT_DIR)

        return jsonify({"success": True, "filename": os.path.basename(output_path)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


# ---------------------------
# UI vs Graph compare endpoint (unchanged)
# ---------------------------
@app.route("/compare/ui_vs_graph", methods=["POST"])
def compare_ui_vs_graph():
    try:
        from graph.neo4j_builder import Neo4jBuilder
        ui_attrs = set(request.json.get("attributes", []))
        neo = Neo4jBuilder()

        with neo.driver.session() as session:
            res = session.run("MATCH (a:Attribute) RETURN a.name AS attr")
            graph_attrs = {r["attr"] for r in res}

        neo.close()

        missing_in_graph = sorted(ui_attrs - graph_attrs)
        extra_in_graph = sorted(graph_attrs - ui_attrs)

        return jsonify({
            "ui_count": len(ui_attrs),
            "graph_count": len(graph_attrs),
            "missing_in_graph": missing_in_graph,
            "extra_in_graph": extra_in_graph
        })

    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Aggregated from graph (unchanged)
# ---------------------------
@app.route("/graph/aggregated", methods=["GET"])
def get_aggregated_from_graph():
    try:
        from graph.neo4j_builder import Neo4jBuilder
        neo = Neo4jBuilder()

        with neo.driver.session() as session:
            result = session.run("""
                MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
                RETURN a.name AS attribute, collect(v.value) AS values
            """)

            rows = []
            attribute_map = {}
            for record in result:
                attribute = record["attribute"]
                values = record["values"]
                attribute_map[attribute] = values
                rows.append([attribute] + values)

            max_len = max((len(v) for v in attribute_map.values()), default=0)
            columns = ["Attribute"] + [f"Value{i+1}" for i in range(max_len)]

        neo.close()
        return jsonify({"columns": columns, "rows": rows}), 200

    except Exception as e:
        print(f"❌ Error fetching graph data: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Rules API Endpoints
# ---------------------------
@app.route("/rules/configuration", methods=["GET"])
def get_configuration_rules():
    global LAST_RULESET, LAST_SKU_MATRIX
    try:
        # If LAST_RULESET is stale or missing, rebuild from current graph SKU matrix
        if not LAST_RULESET:
            # try to load from graph first
            current_from_graph = load_sku_matrix_from_graph()
            if current_from_graph:
                LAST_SKU_MATRIX = current_from_graph
                LAST_RULESET = generate_rules_from_sku_matrix(LAST_SKU_MATRIX)
            else:
                if LAST_SKU_MATRIX:
                    LAST_RULESET = generate_rules_from_sku_matrix(LAST_SKU_MATRIX)
                else:
                    return jsonify({"rules": [], "message": "No SKU data available to generate rules."}), 200

        return jsonify(LAST_RULESET), 200

    except Exception as e:
        print(f"❌ Error fetching rules: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rules/recompute", methods=["POST"])
def recompute_rules():
    global LAST_RULESET, LAST_SKU_MATRIX
    try:
        payload = request.get_json(silent=True) or {}
        sku_matrix = payload.get("sku_matrix") or LAST_SKU_MATRIX
        if not sku_matrix:
            # as fallback, try to load from graph
            sku_matrix = load_sku_matrix_from_graph()
            if sku_matrix:
                LAST_SKU_MATRIX = sku_matrix

        if not sku_matrix:
            return jsonify({"error": "No sku_matrix provided and no previous data available."}), 400

        min_confidence = float(payload.get("min_confidence", 0.85))
        min_row_count = int(payload.get("min_row_count", 3))
        min_support_pct = float(payload.get("min_support_pct", 0.0))
        max_examples = int(payload.get("max_examples", 5))
        persist_json = bool(payload.get("persist_json", False))

        rule_res = generate_rules_from_sku_matrix(
            sku_matrix,
            min_confidence=min_confidence,
            min_row_count=min_row_count,
            min_support_pct=min_support_pct,
            max_examples=max_examples,
            persist_json=persist_json
        )
        LAST_SKU_MATRIX = sku_matrix
        LAST_RULESET = rule_res

        return jsonify(rule_res), 0o200
    except Exception as e:
        print(f"❌ Error recomputing rules: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/rules/<rule_id>/preview", methods=["GET"])
def preview_rule(rule_id):
    global LAST_RULESET
    try:
        if not LAST_RULESET:
            return jsonify({"error": "No rules available"}), 404
        found = next((r for r in LAST_RULESET.get("rules", []) if r["id"] == rule_id), None)
        if not found:
            return jsonify({"error": "Rule not found"}), 404
        return jsonify({"examples": found.get("examples", []), "rule": found}), 200
    except Exception as e:
        print(f"❌ Error previewing rule: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Rebuild SKU matrix endpoint (NEW improved)
# ---------------------------
@app.route("/graph/rebuild_sku_matrix", methods=["GET"])
def rebuild_sku_matrix_from_graph():
    """
    Rebuilds per-SKU matrix based on Neo4j SKU nodes and their HAS_VALUE relationships.
    Returns: {"sku_matrix": [ {sku, attributes:[[attr,val], ...]}, ... ] }
    """
    try:
        sku_matrix = load_sku_matrix_from_graph()
        return jsonify({"sku_matrix": sku_matrix}), 200
    except Exception as e:
        print(f"Error rebuilding sku matrix from graph: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
