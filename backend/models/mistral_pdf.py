

# import os
# import re
# import pdfplumber
# import pytesseract
# import ollama
# import pandas as pd
# from pdf2image import convert_from_path
# from PIL import Image, ImageEnhance, ImageFilter
# from collections import defaultdict

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# MODEL_NAME = "mistral"
# POPPLER_PATH = r"C:\Users\chetu\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
# MAX_CHARS_PER_CHUNK = 3000


# # ----------------------------
# # 1️⃣ EXTRACT TEXT FROM PDF (with OCR fallback)
# # ----------------------------
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_file) as pdf:
#             for page_num, page in enumerate(pdf.pages, 1):
#                 try:
#                     # Try table extraction
#                     tables = page.extract_tables() or []
#                     for table in tables:
#                         for row in table:
#                             if row:
#                                 text += " | ".join([str(cell).strip() for cell in row if cell]) + "\n"

#                     # Regular text extraction
#                     page_text = page.extract_text()
#                     if page_text:
#                         text += page_text + "\n"
#                     else:
#                         print(f"⚠️ No text found on page {page_num}")
#                 except Exception as e:
#                     print(f"⚠️ Error reading page {page_num}: {e}")
#     except Exception as e:
#         print(f"❌ Failed to open PDF: {e}")

#     cleaned_text = re.sub(r'\s+', ' ', text).strip()
   

#     # OCR fallback
#     if not cleaned_text:
#         print("🧾 Running OCR (image-based PDF detected)...")
#         try:
#             images = convert_from_path(pdf_file, poppler_path=POPPLER_PATH)
#             ocr_text = ""
#             for i, image in enumerate(images):
#                 print(f"🖼️ Processing page {i + 1}/{len(images)} for OCR...")

#                 img = image.convert("L")  # grayscale
#                 img = img.filter(ImageFilter.MedianFilter())
#                 enhancer = ImageEnhance.Contrast(img)
#                 img = enhancer.enhance(2)
#                 img = img.point(lambda x: 0 if x < 140 else 255, '1')

#                 ocr_page_text = pytesseract.image_to_string(img, lang='eng')
#                 ocr_text += ocr_page_text + "\n"

#             cleaned_text = re.sub(r'\s+', ' ', ocr_text).strip()
#             if cleaned_text:
#                 print("✅ OCR extraction successful.")
#             else:
#                 print("⚠️ OCR could not extract readable text.")
#         except Exception as e:
#             print(f"❌ OCR error: {e}")

#     if not cleaned_text:
#         print("⚠️ No extractable text found in this PDF.")
#     return cleaned_text


# # ----------------------------
# # 2️⃣ SPLIT TEXT INTO CHUNKS
# # ----------------------------
# def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
#     if not text:
#         return []
#     chunks = []
#     while len(text) > 0:
#         chunk = text[:max_chars]
#         split_pos = max(chunk.rfind("."), chunk.rfind("\n"))
#         if split_pos > 100:
#             chunk = chunk[:split_pos + 1]
#         chunks.append(chunk.strip())
#         text = text[len(chunk):].strip()
#     return chunks


# # ----------------------------
# # 3️⃣ CALL MISTRAL MODEL
# # ----------------------------
# def extract_attributes(text_chunk, domain_prompt):
#     if not text_chunk.strip():
#         return ""
#     prompt = f"""{domain_prompt}

# You are a senior automotive product intelligence analyst. Your task is to analyze the following brochure text and extract all meaningful, structured product attributes and their possible values.

# The text may contain descriptive marketing content, variant details, specifications, and feature highlights. Ignore all marketing or aesthetic language, and focus only on factual, technical, or categorical information that defines the product.

# Your objective is to produce a clean, structured list of attributes and their corresponding values that describe the vehicle comprehensively.

# Follow these detailed instructions:



# 1. **Include only meaningful product attributes** related to:
#    - **Engine & Performance:** Engine Type, Displacement, Power, Torque, Transmission Type, Drivetrain, Fuel Type, Hybrid System Type, Driving Modes, Battery Capacity, Motor Power, Mileage, Emission Standard.
#    - **Dimensions & Weight:** Length, Width, Height, Wheelbase, Ground Clearance, Boot Space, Turning Radius, Kerb Weight.
#    - **Exterior Features:** Headlamp Type, DRLs, Fog Lamps, Roof Rails, Alloy Wheel Size, Tyre Size, Tail Lamp Type, Door Handles, Paint Options.
#    - **Interior & Comfort:** Seat Material, Upholstery Color, Seat Configuration (e.g., 6-seater / 7-seater), Infotainment Screen Size, Ambient Lighting, Steering Controls, Climate Control Type, Cruise Control, Keyless Entry, Start/Stop Button, AC Type, Sunroof Type.
#    - **Safety & Driver Assistance:** Airbags, ABS, EBD, ESP, Hill Hold Assist, ADAS, Traction Control, ISOFIX, Parking Sensors, Camera Type, Speed Alert System, Immobilizer.
#    - **Connectivity & Infotainment:** Touchscreen Display Size, Speaker Count, Audio System Brand, Android Auto / Apple CarPlay Support, USB Ports, Wireless Charging, Connected Car Features.
#    - **Variants & Trims:** Variant Names, Edition, Transmission Options, Seating Layouts, Trim Levels, Special Editions.
#    - **Electrical / Hybrid Attributes (for HEV/EV models):** Motor Type, Hybrid Type (Mild / Strong / Plug-in), Battery Capacity, Battery Type, Regenerative Braking, Drive Modes, EV Range, Charging Type, Charging Time.
#    - **Warranty & Maintenance:** Standard Warranty, Extended Warranty, Service Interval, Roadside Assistance.
#    - **Price & Launch:** Launch Year, Ex-Showroom Price, On-Road Price, Booking Amount (if mentioned).

# 2. **Normalization Rules:**
#    - Expand abbreviations (e.g., MT → Manual Transmission, AT → Automatic Transmission, CVT → Continuously Variable Transmission).
#    - Normalize unit expressions (e.g., “2L” → “2.0L”, “bhp” → “BHP”).
#    - Combine all unique values of the same attribute across variants into one line.
#    - Remove duplicates, special characters, or promotional phrases.
#    - Do not include sentences, bullet points, or marketing slogans.
#    - Focus only on measurable or categorical specifications.

# 3. **Your output should be clear and concise.**  
#    Each attribute must appear only once with all its possible values.
   
# 4. **If multiple trims or variants are mentioned, combine their differing values.**  
#    Example:
#    Transmission = Manual, Automatic, e-CVT
#    Seat Configuration = 6-seater, 7-seater

# 5. **If some attribute values are unclear or partially stated, infer meaning from nearby text context (e.g., “Smart Hybrid Technology” → Hybrid Type = Strong Hybrid).**

# Now, analyze the following extracted brochure text carefully and output the structured attributes in the required format.




# Output format:
# Attribute = Value1, Value2, Value3

# Example:
# Fuel Type = Petrol, Diesel, CNG, Electric
# Transmission = Manual, Automatic
# Color = Red, Blue, White 

# Text:
# {text_chunk}
# """
#     try:
#         response = ollama.chat(
#             model=MODEL_NAME,
#             messages=[{'role': 'user', 'content': prompt}]
#         )
#         return response["message"]["content"].strip()
#     except Exception as e:
#         print(f"⚠️ Mistral error: {e}")
#         return ""


# # ----------------------------
# # 4️⃣ PARSE MODEL OUTPUT
# # ----------------------------
# def parse_attributes(raw_text):
#     attr_dict = defaultdict(list)
#     if not raw_text:
#         return attr_dict
#     lines = re.split(r'[\n;]', raw_text)
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         parts = re.split(r'(?<!\w)=(?!\w)', line)
#         if len(parts) >= 2:
#             attr = parts[0].strip()
#             values = [v.strip() for v in re.split(r"[;,|]", parts[1]) if v.strip()]
#             attr_dict[attr].extend(values)
#     return attr_dict


# # ----------------------------
# # 5️⃣ MERGE MULTIPLE CHUNKS
# # ----------------------------
# def merge_attributes(list_of_dicts):
#     merged = defaultdict(list)
#     for d in list_of_dicts:
#         for attr, vals in d.items():
#             merged[attr].extend(vals)
#     for attr in merged:
#         merged[attr] = list(dict.fromkeys(merged[attr]))  # remove duplicates
#     return merged


# # ----------------------------
# # 6️⃣ MAIN WRAPPER FUNCTION
# # ----------------------------
# def process_pdf_with_mistral(pdf_path, domain_prompt):
#     print(f"📘 Extracting text from PDF: {pdf_path}")
#     pdf_text = extract_text_from_pdf(pdf_path)

#     if not pdf_text:
#         return {"columns": [], "rows": []}

#     print("📝 Splitting text into chunks...")
#     chunks = chunk_text(pdf_text)
#     print(f"🔍 Extracting attributes using Mistral... ({len(chunks)} chunks found)")

#     all_attrs = []
#     for i, chunk in enumerate(chunks, 1):
#         print(f"→ Processing chunk {i}/{len(chunks)}...")
#         raw_output = extract_attributes(chunk, domain_prompt)
#         parsed = parse_attributes(raw_output)
#         if parsed:
#             all_attrs.append(parsed)

#     merged_attr = merge_attributes(all_attrs)

#     if not merged_attr:
#         print("⚠️ No attributes found in PDF.")
#         return {"columns": [], "rows": []}

#     # Prepare table structure for frontend
#     max_values = max(len(v) for v in merged_attr.values())
#     columns = ["Attribute"] + [f"Value{i + 1}" for i in range(max_values)]
#     rows = []
#     for attr, vals in merged_attr.items():
#         row = [attr] + vals + [""] * (max_values - len(vals))
#         rows.append(row)

#     print("✅ PDF extraction complete.")
#     return {"columns": columns, "rows": rows}





import os
import re
import pdfplumber
import pytesseract
import ollama
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from collections import defaultdict

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_NAME = "mistral"
POPPLER_PATH = r"C:\Users\chetu\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
MAX_CHARS_PER_CHUNK = 3000


# ----------------------------
# 1️⃣ Extract text from PDF (with OCR fallback)
# ----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        if row:
                            text += " | ".join([str(cell).strip() for cell in row if cell]) + "\n"

                # Extract normal text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ PDF read error: {e}")

    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    # OCR fallback
    if not cleaned_text:
        print("🧾 Running OCR (image-based PDF detected)...")
        try:
            images = convert_from_path(pdf_file, poppler_path=POPPLER_PATH)
            ocr_text = ""
            for i, image in enumerate(images):
                img = image.convert("L")
                img = img.filter(ImageFilter.MedianFilter())
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2)
                img = img.point(lambda x: 0 if x < 140 else 255, '1')
                ocr_text += pytesseract.image_to_string(img, lang='eng') + "\n"
            cleaned_text = re.sub(r'\s+', ' ', ocr_text).strip()
        except Exception as e:
            print(f"⚠️ OCR error: {e}")

    return cleaned_text


# ----------------------------
# 2️⃣ Split text into chunks
# ----------------------------
def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    chunks = []
    while len(text) > 0:
        chunk = text[:max_chars]
        split_pos = max(chunk.rfind("."), chunk.rfind("\n"))
        if split_pos > 100:
            chunk = chunk[:split_pos + 1]
        chunks.append(chunk.strip())
        text = text[len(chunk):].strip()
    return chunks


# ----------------------------
# 3️⃣ Mistral Extraction
# ----------------------------
def extract_with_mistral(text_chunk, domain_prompt):
    if not text_chunk.strip():
        return ""

    prompt = f"""
{domain_prompt}

You are a senior automotive product intelligence analyst. Your task is to analyze the following brochure text and extract all factual, structured product attributes and their possible values.

The text may contain descriptive marketing content, variant details, specifications, and feature highlights. Ignore all marketing or aesthetic language, and focus only on technical or categorical information explicitly stated in the brochure.

Your objective is to produce a clean, structured list of attributes and their corresponding values that describe the vehicle accurately.

Follow these detailed instructions:

1. **Output format (strictly follow this):**
   Attribute = Value1, Value2, Value3

   Example:
   Fuel Type = Petrol, Diesel
   Transmission = Manual, Automatic
   Color Options = Silver, White, Black

2. **Include only meaningful product attributes** related to:
   - **General Info:** Make, Model, Variant Name, Launch Year, Body Type, Segment.
   - **Engine & Performance:** Engine Type, Displacement, Power, Torque, Transmission Type, Drivetrain, Fuel Type, Hybrid System Type, Driving Modes, Mileage.
   - **Dimensions & Weight:** Length, Width, Height, Wheelbase, Ground Clearance, Boot Space, Turning Radius, Kerb Weight.
   - **Exterior Features:** Headlamp Type, DRLs, Fog Lamps, Alloy Wheel Size, Tyre Size, Roof Rails, Tail Lamp Type, Paint Options.
   - **Interior & Comfort:** Seat Material, Upholstery Color, Infotainment Screen Size, Climate Control Type, Cruise Control, Keyless Entry, Start/Stop Button, Sunroof Type.
   - **Safety & Driver Assistance:** Airbags, ABS, EBD, ESP, Hill Hold Assist, Traction Control, ISOFIX, Parking Sensors, Camera Type.
   - **Connectivity & Infotainment:** Display Size, Speaker Count, Audio System Brand, Android Auto / Apple CarPlay, USB Ports, Wireless Charging.
   - **Variants & Trims:** Variant Names, Edition, Transmission Options, Seating Layouts.
   - **Electrical/Hybrid:** Motor Type, Hybrid Type, Battery Capacity, EV Range, Charging Type, Charging Time.
   - **Warranty & Maintenance:** Standard Warranty, Extended Warranty, Service Interval.
   - **Price & Launch:** Ex-Showroom Price, On-Road Price, Launch Year.

3. **Rules:**
   - ✅ Include only attributes whose values are **explicitly stated** in the text.
   - ❌ Do NOT write values such as “Not specified”, “Not available”, “Unspecified”, “Assuming”, “Likely”, “Estimated”, “Unknown”, or any explanation of what the model assumes.
   - ❌ Do NOT guess or infer any value from model design, segment, or name.
   - ✅ If an attribute has no explicit value in the text, **omit that entire attribute**.
   - ✅ Combine all unique values for the same attribute across variants.
   - ✅ Expand abbreviations (MT → Manual Transmission, AT → Automatic Transmission, etc.).
   - ✅ Normalize units (2L → 2.0L, bhp → BHP).
   - ✅ Use simple, factual phrasing for values.

4. **Formatting rules:**
   - Each attribute appears once.
   - Each value should be separated by a comma and a space.
   - No bullet points, colons, or extra commentary.
   - Output only the final list, with no additional explanation or context.

Now analyze the text carefully and output only the factual structured attributes in the required format.

Output format:
Attribute = Value1, Value2, Value3


Text:
{text_chunk}
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Mistral error: {e}")
        return ""


# ----------------------------
# 4️⃣ Normalizer (Python)
# ----------------------------
# def normalize_output(text):
#     if not text.strip():
#         return ""

#     lines = text.split("\n")
#     clean_lines = []

#     for line in lines:
#         line = line.strip()
#         # Remove numbering or bullets
#         line = re.sub(r"^[\d\-\•\*\.\)\s]+", "", line)
#         # Replace colons with equals
#         line = re.sub(r":", "=", line)
#         # Normalize spacing
#         line = re.sub(r"\s+=\s+", " = ", line)

#         if not re.search(r"=", line):
#             continue

#         parts = line.split("=", 1)
#         if len(parts) < 2:
#             continue

#         attr = re.sub(r"[^A-Za-z0-9\s\-/]", "", parts[0]).strip().title()
#         vals = re.sub(r"[^A-Za-z0-9\s,\-/]", "", parts[1]).strip()

#         attr = re.sub(r"\s+", " ", attr)
#         vals = re.sub(r"\s+", " ", vals)

#         clean_lines.append(f"{attr} = {vals}")

#     return "\n".join(list(dict.fromkeys(clean_lines)))  # Deduplicate

# def normalize_output(text):
#     if not text.strip():
#         return ""

#     lines = text.split("\n")
#     clean_lines = []

#     for line in lines:
#         line = line.strip()
#         line = re.sub(r"^[\d\-\•\*\.\)\s]+", "", line)
#         line = re.sub(r":", "=", line)
#         line = re.sub(r"\s+=\s+", " = ", line)

#         if not re.search(r"=", line):
#             continue

#         parts = line.split("=", 1)
#         if len(parts) < 2:
#             continue

#         attr = re.sub(r"[^A-Za-z0-9\s\-/]", "", parts[0]).strip().title()
#         vals = re.sub(r"[^A-Za-z0-9\s,\-/]", "", parts[1]).strip()

#         # 🚫 Skip generic or useless values
#         if not vals or vals.lower() in [
#             "not specified", "n a", "na", "none", "available",
#             "optional", "standard", "as applicable", "--", "depends"
#         ]:
#             continue

#         attr = re.sub(r"\s+", " ", attr)
#         vals = re.sub(r"\s+", " ", vals)

#         clean_lines.append(f"{attr} = {vals}")

#     # Remove duplicate lines
#     return "\n".join(list(dict.fromkeys(clean_lines)))


def normalize_output(text):
    if not text.strip():
        return ""

    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()
        line = re.sub(r"^[\d\-\•\*\.\)\s]+", "", line)
        line = re.sub(r":", "=", line)
        line = re.sub(r"\s+=\s+", " = ", line)

        if not re.search(r"=", line):
            continue

        parts = line.split("=", 1)
        if len(parts) < 2:
            continue

        attr = re.sub(r"[^A-Za-z0-9\s\-/]", "", parts[0]).strip().title()
        vals = re.sub(r"[^A-Za-z0-9\s,\-/]", "", parts[1]).strip()

        # 🚫 Skip uncertain or placeholder values
        if not vals or any(word in vals.lower() for word in [
            "not specified", "unspecified", "not provided", "not available",
            "assuming", "assumed", "unknown", "n/a", "depends", "based on",
            "interpreted", "likely", "estimate", "probable", "not stated in text", "not explicitly stated"
        ]):
            continue

        attr = re.sub(r"\s+", " ", attr)
        vals = re.sub(r"\s+", " ", vals)

        clean_lines.append(f"{attr} = {vals}")

    return "\n".join(list(dict.fromkeys(clean_lines)))



# ----------------------------
# 5️⃣ Parse & Merge
# ----------------------------
def parse_attributes(raw_text):
    attr_dict = defaultdict(list)
    lines = re.split(r'[\n;]', raw_text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'(?<!\w)=(?!\w)', line)
        if len(parts) >= 2:
            attr = parts[0].strip()
            values = [v.strip() for v in re.split(r"[;,|]", parts[1]) if v.strip()]
            attr_dict[attr].extend(values)
    return attr_dict


def merge_attributes(list_of_dicts):
    merged = defaultdict(list)
    for d in list_of_dicts:
        for attr, vals in d.items():
            merged[attr].extend(vals)
    for attr in merged:
        merged[attr] = list(dict.fromkeys(merged[attr]))
    return merged


# ----------------------------
# 6️⃣ Main Mistral + Normalizer Pipeline
# ----------------------------
def process_pdf_with_mistral_normalizer(pdf_path, domain_prompt):
    print(f"📘 Extracting text from PDF: {pdf_path}")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return {"columns": [], "rows": []}

    chunks = chunk_text(pdf_text)
    print(f"🧠 Processing {len(chunks)} chunks using Mistral + Normalizer...")

    all_attrs = []
    for i, chunk in enumerate(chunks, 1):
        print(f"→ Chunk {i}/{len(chunks)} | Extracting with Mistral...")
        mistral_output = extract_with_mistral(chunk, domain_prompt)

        print("→ Normalizing output...")
        normalized_output = normalize_output(mistral_output)

        parsed = parse_attributes(normalized_output)
        if parsed:
            all_attrs.append(parsed)

    merged = merge_attributes(all_attrs)
    if not merged:
        print("⚠️ No attributes found in PDF.")
        return {"columns": [], "rows": []}

    max_values = max(len(v) for v in merged.values())
    columns = ["Attribute"] + [f"Value{i + 1}" for i in range(max_values)]
    rows = [[attr] + vals + [""] * (max_values - len(vals)) for attr, vals in merged.items()]

    print("✅ Mistral + Normalizer extraction complete.")
    return {"columns": columns, "rows": rows}
