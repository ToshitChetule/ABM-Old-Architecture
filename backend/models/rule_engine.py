# models/rule_engine.py
import json
import uuid
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

# Default attributes to ignore because they are universal or not useful for config logic.
DEFAULT_ATTRIBUTE_BLACKLIST = {
    "Product family",
    "Product name",
    "Note",
    "Model",  # example
    ""       # blank attribute names
}


def is_strict_rule_valid(a: str, val_a: str, b: str, val_b: str, rows: List[Dict[str,str]]) -> bool:
    """
    Strict validation:
    A → B is valid ONLY IF:
      For every row where A = val_a, B MUST equal val_b.
    Any violation invalidates the rule.
    """
    for r in rows:
        if r.get(a) == val_a:
            if r.get(b) != val_b:  # violation detected
                return False
    return True


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def _row_to_map(sku_row: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert per-SKU {"sku": "...", "attributes": [[attr, val], ...]} into a dict {attr: val}
    """
    res = {}
    for pair in sku_row.get("attributes", []) if sku_row else []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            a = _normalize_text(pair[0])
            v = _normalize_text(pair[1])
            if a:
                res[a] = v
    return res

def _human_rule_text(if_map: Dict[str, str], then_map: Dict[str, str]) -> str:
    left = " AND ".join([f"{k} = {v}" for k, v in if_map.items()])
    right = " AND ".join([f"{k} = {v}" for k, v in then_map.items()])
    if not left:
        return f"THEN {right}"
    if not right:
        return f"IF {left}"
    return f"IF {left} THEN {right}"

def save_rules_to_json(ruleset: Dict[str, Any], path: str) -> None:
    """
    Persist ruleset to JSON file.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ruleset, fh, indent=2, ensure_ascii=False)

def generate_rules_from_sku_matrix(
    sku_matrix: List[Dict[str, Any]],
    min_confidence: float = 0.85,
    min_row_count: int = 3,
    min_support_pct: float = 0.0,
    max_examples: int = 1,
    persist_json: bool = False,
    attribute_blacklist: Optional[set] = None
) -> Dict[str, Any]:
    """
    Deterministic rule generation.

    Parameters:
      - sku_matrix: list of {"sku": "...", "attributes": [[attr, val], ...]}
      - min_confidence: threshold for confidence (0..1)
      - min_row_count: minimum number of rows that must contain the IF antecedent (absolute)
      - min_support_pct: minimum support as fraction of total rows (0..1)
      - max_examples: how many example SKU texts to include per rule
      - persist_json: if True, save to outputs/rules_YYYYMMDD_HHMMSS.json
      - attribute_blacklist: optional set of attribute names to ignore

    Returns:
      {
        "generated_at": "...",
        "total_rows": N,
        "rules": [ {id, if, then, support, support_pct, confidence, examples, rule_text}, ... ]
      }
    """
    if attribute_blacklist is None:
        attribute_blacklist = set(DEFAULT_ATTRIBUTE_BLACKLIST)

    total_rows = len(sku_matrix or [])
    if total_rows == 0:
        return {"generated_at": datetime.utcnow().isoformat() + "Z", "total_rows": 0, "rules": []}

    # 1) Convert each SKU row to map {attr: val}
    rows = []
    for s in sku_matrix:
        rows.append(_row_to_map(s))

    # 2) Build attribute -> value -> list(row_index)
    attr_val_rows: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    attr_counts: Dict[str, int] = Counter()

    for i, rmap in enumerate(rows):
        for attr, val in rmap.items():
            if not val:
                # skip empty values
                continue
            if attr in attribute_blacklist:
                continue
            attr_val_rows[attr][val].append(i)
            attr_counts[attr] += 1

    # 3) Candidate antecedents and consequents: attributes with at least 2 distinct values and not blacklisted
    candidate_attrs = [a for a in attr_val_rows.keys() if len(attr_val_rows[a]) >= 1]

    rules = []

    # 4) Generate pairwise rules A=value_a -> B=value_b
    # We'll compute support = count(A=value_a AND B=value_b)
    # confidence = support / count(A=value_a)
    # filter by thresholds
    min_support_count = max(1, int(min_support_pct * total_rows)) if min_support_pct else 0

    for a in candidate_attrs:
        for val_a, rows_a in attr_val_rows[a].items():
            count_a = len(rows_a)
            # require minimum rows with antecedent
            if count_a < min_row_count:
                continue

            # For performance: compute co-occurrence with all other attributes by scanning rows_a
            co_counter: Dict[Tuple[str, str], int] = Counter()
            for ridx in rows_a:
                rmap = rows[ridx]
                for b, val_b in rmap.items():
                    if b == a:
                        continue
                    if not val_b:
                        continue
                    if b in attribute_blacklist:
                        continue
                    co_counter[(b, val_b)] += 1

            # Evaluate candidate consequents
            for (b, val_b), support_count in co_counter.items():
                # support_count is absolute rows where A=val_a and B=val_b
                if support_count < 1:
                    continue
                if support_count < min_support_count:
                    continue

                confidence = support_count / float(count_a)
                if confidence < min_confidence:
                    continue

                # STRICT RULE VALIDATION — absolutely no violations allowed
                if not is_strict_rule_valid(a, val_a, b, val_b, rows):
                    # If even ONE SKU where A = val_a but B != val_b → rule is invalid
                    continue



                # Avoid trivial rule where attribute names are identical or same attribute on both sides
                if a.lower().strip() == b.lower().strip():
                    continue

                # Build examples (SKU texts) - prefer original SKU string if present
                examples = []
                for ridx in rows_a:
                    if len(examples) >= max_examples:
                        break
                    rmap = rows[ridx]
                    # include only rows that also have b=val_b
                    if rmap.get(b, "") == val_b:
                        # try to find original SKU text from sku_matrix if present
                        original_sku_text = ""
                        try:
                            original_sku_text = _normalize_text(sku_matrix[ridx].get("sku", "") or sku_matrix[ridx].get("SKU", ""))
                        except Exception:
                            original_sku_text = ""
                        if not original_sku_text:
                            # fallback to a compact representation of the row
                            original_sku_text = ", ".join([f"{k}:{v}" for k, v in rmap.items() if v])
                        examples.append(original_sku_text)

                # create rule
                rule_obj = {
                    "id": str(uuid.uuid4()),
                    "if": {a: val_a},
                    "then": {b: val_b},
                    "support": support_count,
                    "support_pct": round((support_count / total_rows) * 100.0, 3),
                    "confidence": round(confidence, 4),
                    "antecedent_row_count": count_a,
                    "examples": examples,
                    "rule_text": _human_rule_text({a: val_a}, {b: val_b})
                }
                rules.append(rule_obj)

    # 5) Post-filtering: remove duplicates (same if→then) and sort by confidence*support (score)
    unique = {}
    for r in rules:
        key = (tuple(sorted(r["if"].items())), tuple(sorted(r["then"].items())))
        # keep rule with higher confidence or higher support
        cur = unique.get(key)
        if not cur:
            unique[key] = r
        else:
            # pick better one
            if (r["confidence"], r["support"]) > (cur["confidence"], cur["support"]):
                unique[key] = r

    deduped_rules = list(unique.values())

    # Sort by confidence desc then support desc
    deduped_rules.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)

    ruleset = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_rows": total_rows,
        "rules": deduped_rules
    }

    if persist_json:
        path = f"outputs/rules_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        save_rules_to_json(ruleset, path)
        ruleset["persisted_to"] = path

    return ruleset
