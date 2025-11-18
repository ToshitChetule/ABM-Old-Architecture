import ollama
import json

# 🧠 The same model you're using elsewhere
MODEL = "llama3"

# 🧩 System prompt: keeps model output structured and predictable
SYSTEM_INSTRUCTIONS = """
You are a product data knowledge-graph refinement assistant.

You receive:
1️⃣ A JSON context about a specific Attribute node in a Neo4j graph
2️⃣ A user prompt describing what refinement to perform

You must reason about the context and respond with a **JSON plan** only.

Rules:
- Do NOT explain or add extra text.
- Return valid JSON, no markdown or prose.
- Include only necessary keys from this list:

{
  "rename_attribute_to": "NewAttributeName",
  "merge_with_attributes": ["OtherAttr1", "OtherAttr2"],
  "keep_values": ["Value1", "Value2"],
  "add_values": ["NewValue1", "NewValue2"],
  "remove_values": ["ValueToRemove"]
}

Guidelines:
- If user says "rename", populate rename_attribute_to.
- If user says "merge", list attributes to merge in merge_with_attributes.
- If user mentions values to keep/add/remove, fill those arrays.
- If not mentioned, omit that key entirely.
- Never include commentary or code fences.
"""


def refine_with_graph_context(graph_context: dict, user_prompt: str) -> dict:
    """
    Sends graph context + user prompt to the LLaMA model and returns
    a structured refinement plan in JSON.
    """

    try:
        # Prepare the model input
        payload = json.dumps({
            "graph_context": graph_context,
            "user_prompt": user_prompt
        }, ensure_ascii=False)

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": payload}
        ]

        # 🧠 Send to LLaMA through Ollama
        response = ollama.chat(model=MODEL, messages=messages)
        raw_output = response["message"]["content"].strip()

        # Attempt to parse JSON output
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Model sometimes adds extra text—try to extract valid JSON block
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end != -1:
                candidate = raw_output[start:end]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        # If parsing still fails, fallback to empty dict
        return {}

    except Exception as e:
        print(f"❌ Error in refine_with_graph_context: {e}")
        return {}
