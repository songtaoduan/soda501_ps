###########################################
# LLM-Assisted Data Extraction PS 7
###########################################


######## Conceptual Questions
# Q2: Human-in-the-loop extraction needs both precision/recall evaluation and spot audits because they detect different types of problems. 
# Precision and recall, measured on a labeled gold dataset, show how accurately the system extracts the correct information overall. 
# However, a small gold set may miss failures on rare or unusual documents that are not included in the dataset. 
# Spot audits—randomly checking outputs—help identify these unexpected errors. On the other hand, auditing only a small random sample 
# might miss systematic errors, such as the model repeatedly failing to extract a particular field, which precision/recall metrics would reveal.


#### Applied Exercises
# see after the end of the demo code (line 424)
# VS code interface still has idention issue that I haven't fixed yet.

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
import os
import json
import re

import numpy as np
import pandas as pd

import torch

import requests

from datetime import date
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

np.random.seed(123)
set_seed(123)

# ---------------------------------------------------------------------
# Part 1: Define a Schema (What fields do we want?)
# ---------------------------------------------------------------------
EventType = Literal[
    "protest",
    "election",
    "policy_change",
    "violence",
    "disaster",
    "other"
]

GeoPrecision = Literal[
    "country_only",
    "admin1_or_state",
    "city_or_local",
    "unknown"
]

class EvidenceSpan(BaseModel):
    field: Literal["event_type", "date", "location", "actors", "outcome"]
    quote: str

class EventExtraction(BaseModel):
    doc_id: str

    # NOTE (teaching/demo setting):
    # For local small models, strict JSON/schema enforcement often yields empty outputs.
    # We therefore provide safe defaults so the pipeline can produce "messy but usable"
    # records while still flagging uncertainty. In a production setting, tighten these
    # requirements and fail fast.

    event_type: EventType = "other"

    event_date_iso: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD if available; otherwise null."
    )
    date_is_approximate: bool = Field(
        default=True,
        description="True if the date is estimated/inferred (e.g., 'early April')."
    )

    country: Optional[str] = None
    admin1_or_state: Optional[str] = None
    city_or_local: Optional[str] = None
    geo_precision: GeoPrecision = "unknown"

    actors: List[str] = Field(default_factory=list, description="Key actors mentioned (individuals, orgs, groups).")

    outcome_summary: Optional[str] = Field(
        default=None,
        description="One-sentence outcome summary (what happened)."
    )

    extraction_confidence: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Model self-rated confidence (0 to 1)."
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="List of issues that make extraction uncertain (e.g., missing date, vague location)."
    )
    evidence: List[EvidenceSpan] = Field(
        default_factory=list,
        description="Short quotes supporting each extracted field (if available)."
    )


# ---------------------------------------------------------------------
# Part 2: Create Messy Text Inputs (Mini Corpus)
# ---------------------------------------------------------------------
docs = [
    {"doc_id": "doc_001", "text": "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested."},
    {"doc_id": "doc_002", "text": "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote."},
    {"doc_id": "doc_003", "text": "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital."},
    {"doc_id": "doc_004", "text": "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents."},
    {"doc_id": "doc_005", "text": "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear."},
    {"doc_id": "doc_006", "text": "The governor announced a new curfew order effective immediately. Critics called it an overreach."},
    {"doc_id": "doc_007", "text": "Early April saw renewed demonstrations in the northern province after fuel prices rose again."},
    {"doc_id": "doc_008", "text": "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said."},
    {"doc_id": "doc_009", "text": "Opposition leaders met with international observers in Brussels to discuss election monitoring."},
    {"doc_id": "doc_010", "text": "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm."},
    {"doc_id": "doc_011", "text": "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility."},
    {"doc_id": "doc_012", "text": "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings."},
    {"doc_id": "doc_013", "text": "Following a landslide, the ministry declared a state of emergency in two districts."},
    {"doc_id": "doc_014", "text": "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision."},
    {"doc_id": "doc_015", "text": "The article mentions reforms and elections in passing but gives no clear time or place."},
]

docs_df = pd.DataFrame(docs)

print("\n------------------------------")
print("Input corpus (first 5 docs)")
print("------------------------------")
print(docs_df.head())
print("docs_df shape:", docs_df.shape)

# ---------------------------------------------------------------------
# Part 3: Prompt Design (Schemas + Guardrails)
# ---------------------------------------------------------------------
json_template = {
    "doc_id": "doc_XXX",
    "event_type": "other",
    "event_date_iso": None,
    "date_is_approximate": False,
    "country": None,
    "admin1_or_state": None,
    "city_or_local": None,
    "geo_precision": "unknown",
    "actors": [],
    "outcome_summary": None,
    "extraction_confidence": 0.5,
    "uncertainty_flags": [],
    "evidence": [
        {"field": "event_type", "quote": ""},
        {"field": "date", "quote": ""},
        {"field": "location", "quote": ""},
        {"field": "actors", "quote": ""},
        {"field": "outcome", "quote": ""}
    ]
}

system_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Return EXACTLY the following 9 lines, one per line, in the format key: value\n"
    "Use empty value if unknown.\n"
    "\n"
    "event_type: protest|election|policy_change|violence|disaster|other\n"
    "event_date_iso: YYYY-MM-DD\n"
    "date_is_approximate: true|false\n"
    "country:\n"
    "admin1_or_state:\n"
    "city_or_local:\n"
    "geo_precision: country_only|admin1_or_state|city_or_local|unknown\n"
    "actors: comma-separated list\n"
    "outcome_summary: one sentence\n"
    "\n"
    "Do not output anything else.\n"
)

# ---------------------------------------------------------------------
# Part 4: Local LLM Structured Extraction (Batch Processing)
# ---------------------------------------------------------------------
# Model choice: small, free, runs on CPU (slow but fine for class demos)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("\n------------------------------")
print("Loading tokenizer + model")
print("------------------------------")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if use_gpu else torch.device("cpu")
model = model.to(device)

print("Model:", model_name)
print("Device:", device)

extractions = []

print("\n------------------------------")
print("Running LOCAL LLM extraction (one doc at a time)")
print("------------------------------")

for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]

    prompt = (
        f"{system_instructions}\n"
        f"Document ID: {doc_id}\n"
        f"Text: {text}\n"
    )

    # 1) Tokenize (explicit)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Generate (explicit)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # 3) Decode (explicit)
    out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 4) Parse labeled key:value lines (best-effort)
    parse_ok = True
    parse_error = ""
    parse_flags: List[str] = []

    allowed_keys = {
        "event_type",
        "event_date_iso",
        "date_is_approximate",
        "country",
        "admin1_or_state",
        "city_or_local",
        "geo_precision",
        "actors",
        "outcome_summary"
    }

    lines = [ln.strip() for ln in out_text.splitlines() if ":" in ln]
    kv = {}

    for ln in lines:
        k, v = ln.split(":", 1)
        k_norm = k.strip().lower()
        if k_norm in allowed_keys:
            kv[k_norm] = v.strip()

    if len(kv) == 0:
        parse_ok = False
        parse_error = "no_key_value_lines_found"
        parse_flags.append("parse_failed_local_model_output")

    event_type = (kv.get("event_type", "other") or "other").strip().lower()
    if event_type not in {"protest", "election", "policy_change", "violence", "disaster", "other"}:
        parse_flags.append("invalid_event_type_from_model")
        event_type = "other"

    event_date_iso = (kv.get("event_date_iso", "") or "").strip() or None

    date_is_approx_raw = (kv.get("date_is_approximate", "") or "").strip().lower()
    if date_is_approx_raw in {"true", "false"}:
        date_is_approximate = (date_is_approx_raw == "true")
    else:
        parse_flags.append("date_is_approximate_missing_or_invalid")
        date_is_approximate = True

    country = (kv.get("country", "") or "").strip() or None
    admin1_or_state = (kv.get("admin1_or_state", "") or "").strip() or None
    city_or_local = (kv.get("city_or_local", "") or "").strip() or None

    geo_precision = (kv.get("geo_precision", "unknown") or "unknown").strip().lower()
    if geo_precision not in {"country_only", "admin1_or_state", "city_or_local", "unknown"}:
        parse_flags.append("invalid_geo_precision_from_model")
        geo_precision = "unknown"

    actors_raw = (kv.get("actors", "") or "").strip()
    actors = [a.strip() for a in actors_raw.split(",") if a.strip()] if actors_raw else []

    outcome_summary = (kv.get("outcome_summary", "") or "").strip() or None

    extracted_obj = EventExtraction(
        doc_id=doc_id,
        event_type=event_type,
        event_date_iso=event_date_iso,
        date_is_approximate=date_is_approximate,
        country=country,
        admin1_or_state=admin1_or_state,
        city_or_local=city_or_local,
        geo_precision=geo_precision,
        actors=actors,
        outcome_summary=outcome_summary,
        extraction_confidence=0.35 if parse_ok else 0.2,
        uncertainty_flags=parse_flags,
        evidence=[]
    )

    extra_dict = extracted_obj.model_dump()


    # 5) Attach trace fields (explicit)
    extra_dict["raw_text"] = text
    extra_dict["local_model_raw_output"] = out_text
    extra_dict["parse_ok"] = parse_ok
    extra_dict["parse_error"] = parse_error

    # 6) Flatten list fields for CSV (explicit)
    extra_dict["evidence_json"] = json.dumps(extra_dict["evidence"], ensure_ascii=False)
    extra_dict["uncertainty_flags_json"] = json.dumps(extra_dict["uncertainty_flags"], ensure_ascii=False)
    extra_dict.pop("evidence")
    extra_dict.pop("uncertainty_flags")

    extractions.append(extra_dict)

# 7) Build dataframe + save
extractions_df = pd.DataFrame(extractions)

print("\n------------------------------")
print("Extracted records (first 5 rows)")
print("------------------------------")
print(extractions_df.head())
print("extractions_df shape:", extractions_df.shape)

os.makedirs("outputs", exist_ok=True)
extractions_df.to_csv("outputs/extractions_raw.csv", index=False)

# ---------------------------------------------------------------------
# Part 5: Uncertainty Checks (Automatic Flags for Human Review)
# ---------------------------------------------------------------------
extractions_df["extraction_confidence"] = pd.to_numeric(extractions_df["extraction_confidence"], errors="coerce")

extractions_df["flag_parse_failed"] = ~extractions_df["parse_ok"]
extractions_df["flag_low_confidence"] = extractions_df["extraction_confidence"] < 0.70
extractions_df["flag_missing_date"] = extractions_df["event_date_iso"].isna()
extractions_df["flag_missing_country"] = extractions_df["country"].isna()
extractions_df["flag_geo_unknown"] = extractions_df["geo_precision"].isin(["unknown", "country_only"])

flag_cols = [
    "flag_parse_failed",
    "flag_low_confidence",
    "flag_missing_date",
    "flag_missing_country",
    "flag_geo_unknown"
]
extractions_df["needs_human_review"] = extractions_df[flag_cols].any(axis=1)

print("\n------------------------------")
print("Review flag counts")
print("------------------------------")
print(extractions_df[flag_cols + ["needs_human_review"]].sum(numeric_only=True))

extractions_df.to_csv("outputs/extractions_with_flags.csv", index=False)

# ---------------------------------------------------------------------
# Part 6: Human Validation / Spot-Audits (Create an Audit Sheet)
# ---------------------------------------------------------------------
audit_random_n = 5
audit_random = extractions_df.sample(n=audit_random_n, random_state=123)

audit_flagged = extractions_df[extractions_df["needs_human_review"]].copy()

audit_sheet = pd.concat([audit_random, audit_flagged], ignore_index=True).drop_duplicates(subset=["doc_id"])
audit_sheet = audit_sheet.sort_values("doc_id").reset_index(drop=True)

audit_sheet["human_is_correct"] = ""
audit_sheet["human_correct_event_type"] = ""
audit_sheet["human_correct_date_iso"] = ""
audit_sheet["human_correct_location"] = ""
audit_sheet["failure_mode"] = ""
audit_sheet["reviewer_notes"] = ""

audit_sheet.to_csv("outputs/human_audit_sheet.csv", index=False)

print("\n------------------------------")
print("Wrote outputs/human_audit_sheet.csv")
print("------------------------------")

# ---------------------------------------------------------------------
# Part 7: Evaluation Patterns (Precision/Recall + Auditing)
# ---------------------------------------------------------------------
gold = pd.DataFrame([
    {"doc_id": "doc_001", "event_type_gold": "protest"},
    {"doc_id": "doc_002", "event_type_gold": "policy_change"},
    {"doc_id": "doc_003", "event_type_gold": "election"},
    {"doc_id": "doc_004", "event_type_gold": "disaster"},
    {"doc_id": "doc_005", "event_type_gold": "violence"},
    {"doc_id": "doc_006", "event_type_gold": "policy_change"},
    {"doc_id": "doc_007", "event_type_gold": "protest"},
    {"doc_id": "doc_008", "event_type_gold": "disaster"},
])

eval_df = gold.merge(extractions_df[["doc_id", "event_type"]], on="doc_id", how="left")
eval_df = eval_df.rename(columns={"event_type": "event_type_pred"})
eval_df["event_type_pred"] = eval_df["event_type_pred"].fillna("MISSING")

print("\n------------------------------")
print("Evaluation table (gold vs predicted)")
print("------------------------------")
print(eval_df)

print("\n------------------------------")
print("Classification report (event_type)")
print("------------------------------")
print(classification_report(eval_df["event_type_gold"], eval_df["event_type_pred"], zero_division=0))



#  Q3: Replace the API call with Ollama.  
#  First, make sure Ollama is running: ollama pull llama3.1:8b (if not already installed, first install ollama in terminal)
import os
import json
import re
import requests
import pandas as pd

# -----------------------------
# Ollama settings (REPORT THESE)
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  

# -----------------------------
# Prompt (REPORT THIS EXACTLY)
# -----------------------------
PROMPT_TEMPLATE = """You are an information extraction system.

Extract ONE event record from the text and return ONLY a single valid JSON object.
The JSON MUST match this schema exactly (keys and types). If unknown, use null (or [] for lists).
Do not include any extra text, markdown, or code fences.

Schema (example with types):
{{
  "doc_id": "string",
  "event_type": "protest|election|policy_change|violence|disaster|other",
  "event_date_iso": "YYYY-MM-DD or null",
  "date_is_approximate": true,
  "country": "string or null",
  "admin1_or_state": "string or null",
  "city_or_local": "string or null",
  "geo_precision": "country_only|admin1_or_state|city_or_local|unknown",
  "actors": ["string", "..."],
  "outcome_summary": "string or null",
  "extraction_confidence": 0.0,
  "uncertainty_flags": ["string", "..."],
  "evidence": [
    {{"field": "event_type|date|location|actors|outcome", "quote": "short supporting quote"}},
    ...
  ]
}}

Document ID: {doc_id}
Text: {text}
"""

def call_ollama_json(prompt: str) -> str:
    """Call Ollama and return the raw text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Optional: reduce randomness for more stable JSON
        "options": {"temperature": 0.0}
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

def extract_first_json_object(s: str) -> str:
    """
    Best-effort: find the first JSON object in a string.
    Some models occasionally prepend/append text despite instructions.
    """
    s_strip = s.strip()

    # If the entire response already looks like JSON
    if s_strip.startswith("{") and s_strip.endswith("}"):
        return s_strip

    # Otherwise search for JSON object boundaries
    start = s.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")

    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1].strip()

    raise ValueError("Could not find a complete JSON object.")
    
# -----------------------------
# Run extraction over all docs
# -----------------------------
os.makedirs("soda501_ps/07_llm_human/outputs", exist_ok=True)

extractions = []
for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]

    prompt = PROMPT_TEMPLATE.format(doc_id=doc_id, text=text)

    parse_ok = True
    parse_error = ""
    flags = []
    raw = ""

    try:
        raw = call_ollama_json(prompt)

        json_str = extract_first_json_object(raw)
        record = json.loads(json_str)

        validated = EventExtraction(**record)
        out = validated.model_dump()

    except Exception as e:
        parse_ok = False
        parse_error = type(e).__name__
        flags.append(f"parse_or_validation_failed:{parse_error}")

        fallback = EventExtraction(
            doc_id=doc_id,
            event_type="other",
            event_date_iso=None,
            date_is_approximate=True,
            country=None,
            admin1_or_state=None,
            city_or_local=None,
            geo_precision="unknown",
            actors=[],
            outcome_summary=None,
            extraction_confidence=0.0,
            uncertainty_flags=flags,
            evidence=[]
        )

        out = fallback.model_dump()

    out["raw_text"] = text
    out["model_raw_output"] = raw
    out["parse_ok"] = parse_ok
    out["parse_error"] = parse_error

    extractions.append(out)


extractions_df = pd.DataFrame(extractions)
extractions_df.to_csv("outputs/extractions_raw.csv", index=False)

print("Wrote outputs/extractions_raw.csv")
print("Model used:", OLLAMA_MODEL)
print("\nExact prompt template used:\n", PROMPT_TEMPLATE)


# Q4: Uncertainty flags + audit sheet (human-in-the-loop)
## Note that we have extractions_raw.csv from Q3

import os
import json
import pandas as pd
from collections import Counter

# ----------------------------
# Config
# ----------------------------
EXTRACTIONS_PATH = "soda501_ps/07_llm_human/outputs/extractions_raw.csv"
AUDIT_SHEET_PATH = "soda501_ps/07_llm_humanoutputs/audit_sheet.csv"

# Mechanical flag thresholds
LOW_CONF_THRESH = 0.60

# ----------------------------
# Helpers
# ----------------------------
def safe_json_loads(x, default):
    if pd.isna(x) or x is None or str(x).strip() == "":
        return default
    try:
        return json.loads(x)
    except Exception:
        return default

def normalize_text(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def make_evidence_string(evidence_list):
    """
    evidence_list: list[dict] like [{"field":"date","quote":"..."}]
    Returns a compact string for audit viewing.
    """
    if not evidence_list:
        return ""
    parts = []
    for ev in evidence_list:
        field = str(ev.get("field", "")).strip()
        quote = str(ev.get("quote", "")).strip()
        if field and quote:
            parts.append(f"[{field}] {quote}")
        elif quote:
            parts.append(quote)
    return " | ".join(parts)

def compute_mechanical_flags(row):
    """
    Creates at least 4 mechanical review flags.
    Returns (flags_list, needs_review_bool)
    """
    flags = []

    # parse_ok / parse_error may exist from your pipeline; handle if absent
    parse_ok = bool(row.get("parse_ok", True))
    if not parse_ok:
        flags.append("parse_failed")

    conf = row.get("extraction_confidence", None)
    try:
        conf_val = float(conf) if conf is not None and conf != "" else None
    except Exception:
        conf_val = None

    if conf_val is None:
        flags.append("confidence_missing")
    elif conf_val < LOW_CONF_THRESH:
        flags.append("low_confidence")

    # Missing date
    date_iso = normalize_text(row.get("event_date_iso", "")).strip()
    if date_iso == "" or date_iso.lower() == "none" or date_iso.lower() == "null":
        flags.append("date_missing")

    # Missing country
    country = normalize_text(row.get("country", "")).strip()
    if country == "" or country.lower() == "none" or country.lower() == "null":
        flags.append("country_missing")

    # Geo precision unknown
    geo_precision = normalize_text(row.get("geo_precision", "")).strip().lower()
    if geo_precision in ["", "unknown", "none", "null"]:
        flags.append("geo_precision_unknown")

    # Actors empty (actors_json preferred if present)
    actors = row.get("_actors_list", [])
    if not actors:
        flags.append("actors_empty")

    needs_review = len(flags) > 0
    return flags, needs_review

# ----------------------------
# 1) Load extractions
# ----------------------------
if not os.path.exists(EXTRACTIONS_PATH):
    raise FileNotFoundError(f"Missing {EXTRACTIONS_PATH}. Run Part 3 first.")

df = pd.read_csv(EXTRACTIONS_PATH)

# Parse list-like columns if present
# Part 3 code often stores JSON as *_json columns; handle both possibilities.
if "actors_json" in df.columns:
    df["_actors_list"] = df["actors_json"].apply(lambda x: safe_json_loads(x, []))
elif "actors" in df.columns:
    df["_actors_list"] = df["actors"].apply(lambda x: safe_json_loads(x, []) if isinstance(x, str) else (x if isinstance(x, list) else []))
else:
    df["_actors_list"] = [[] for _ in range(len(df))]

if "evidence_json" in df.columns:
    df["_evidence_list"] = df["evidence_json"].apply(lambda x: safe_json_loads(x, []))
elif "evidence" in df.columns:
    df["_evidence_list"] = df["evidence"].apply(lambda x: safe_json_loads(x, []) if isinstance(x, str) else (x if isinstance(x, list) else []))
else:
    df["_evidence_list"] = [[] for _ in range(len(df))]

# ----------------------------
# 2) Create mechanical flags
# ----------------------------
flags_out = []
needs_review_out = []
evidence_out = []
actors_out = []

for _, row in df.iterrows():
    # Make evidence + actors human-readable
    actors_list = row.get("_actors_list", [])
    actors_out.append(", ".join([str(a) for a in actors_list]) if actors_list else "")

    evidence_list = row.get("_evidence_list", [])
    evidence_out.append(make_evidence_string(evidence_list))

    flags, needs_review = compute_mechanical_flags(row)
    flags_out.append(";".join(flags))
    needs_review_out.append(needs_review)

df["mechanical_flags"] = flags_out
df["needs_review"] = needs_review_out
df["actors_str"] = actors_out
df["evidence_quotes"] = evidence_out

# ----------------------------
# 3) Build audit sheet (tutorial-style)
#    Required:
#      (a) raw text
#      (b) extracted fields
#      (c) evidence quotes
#      (d) blank columns for human corrections + failure-mode tags
# ----------------------------

# Raw text column name might differ; handle common names
raw_text_col = "raw_text" if "raw_text" in df.columns else ("text" if "text" in df.columns else None)
if raw_text_col is None:
    df["raw_text"] = ""
    raw_text_col = "raw_text"

audit_cols = [
    "doc_id",
    raw_text_col,
    "event_type",
    "event_date_iso",
    "country",
    "admin1_or_state",
    "city_or_local",
    "geo_precision",
    "actors_str",
    "outcome_summary",
    "extraction_confidence",
    "mechanical_flags",
    "needs_review",
    "evidence_quotes",
]

# Keep only columns that exist (robust to schema choices)
audit_cols = [c for c in audit_cols if c in df.columns]

audit = df[audit_cols].copy()
audit = audit.rename(columns={raw_text_col: "raw_text"})

# Add blank human correction columns + failure tags
audit["human_correct"] = ""  
audit["failure_mode_tags"] = "" 
audit["human_event_type"] = ""
audit["human_event_date_iso"] = ""
audit["human_country"] = ""
audit["human_admin1_or_state"] = ""
audit["human_city_or_local"] = ""
audit["human_geo_precision"] = ""
audit["human_actors"] = ""
audit["human_outcome_summary"] = ""
audit["human_notes"] = ""

# ----------------------------
# 4) Pre-fill at least 5 docs for auditing
#    Strategy: prioritize "needs_review" rows, then fill remaining from top.
# ----------------------------
needs = audit[audit["needs_review"] == True]
if len(needs) >= 5:
    audit_idx = needs.index[:5]
else:
    audit_idx = list(needs.index) + list(audit.index.difference(needs.index)[: (5 - len(needs))])

audit["audit_me"] = ""
audit.loc[audit_idx, "audit_me"] = "YES"  

# Write audit sheet
os.makedirs("outputs", exist_ok=True)
audit.to_csv(AUDIT_SHEET_PATH, index=False)
print(f"Wrote audit sheet template to: {AUDIT_SHEET_PATH}")
print("Fill at least the rows with audit_me == YES (>=5 rows). Then re-run stats section below.")

# ----------------------------
# 5) Audit statistics 
#    Required:
#      (a) share of audited rows marked correct
#      (b) most common failure mode (frequency table)
# ----------------------------
def compute_audit_stats(audit_sheet_path=AUDIT_SHEET_PATH):
    a = pd.read_csv(audit_sheet_path)

    audited = a[a.get("audit_me", "").astype(str).str.upper() == "YES"].copy()
    if audited.empty:
        print("No audited rows found (audit_me == YES).")
        return

    # Share correct: interpret human_correct as TRUE/FALSE/1/0
    def to_bool(x):
        s = str(x).strip().lower()
        if s in ["true", "t", "1", "yes", "y"]:
            return True
        if s in ["false", "f", "0", "no", "n", ""]:
            return False
        return False

    audited["human_correct_bool"] = audited.get("human_correct", "").apply(to_bool)
    share_correct = audited["human_correct_bool"].mean()

    # Failure mode frequency table (split by comma/semicolon)
    fm_counter = Counter()
    for tags in audited.get("failure_mode_tags", "").fillna(""):
        parts = [p.strip() for p in str(tags).replace(";", ",").split(",") if p.strip()]
        fm_counter.update(parts)

    fm_table = pd.DataFrame(fm_counter.most_common(), columns=["failure_mode", "count"])

    print("\n=== Audit statistics ===")
    print(f"Audited rows: {len(audited)}")
    print(f"Share marked correct: {share_correct:.3f}")

    print("\nMost common failure modes:")
    if fm_table.empty:
        print("(none recorded)")
    else:
        print(fm_table.head(20).to_string(index=False))



# Q5: Evaluation + prompt iteration.
import os
import json
import pandas as pd

from sklearn.metrics import classification_report, f1_score, accuracy_score

# =========================================================
# Part 5: Evaluation + Prompt Iteration (event_type)
# =========================================================

# -----------------------------
# Gold set (from tutorial style)
# -----------------------------
gold = pd.DataFrame([
    {"doc_id": "doc_001", "event_type_gold": "protest"},
    {"doc_id": "doc_002", "event_type_gold": "policy_change"},
    {"doc_id": "doc_003", "event_type_gold": "election"},
    {"doc_id": "doc_004", "event_type_gold": "disaster"},
    {"doc_id": "doc_005", "event_type_gold": "violence"},
    {"doc_id": "doc_006", "event_type_gold": "policy_change"},
    {"doc_id": "doc_007", "event_type_gold": "protest"},
    {"doc_id": "doc_008", "event_type_gold": "disaster"},
])

EVENT_TYPES = ["protest", "election", "policy_change", "violence", "disaster", "other"]


# -----------------------------
# Two prompts
# -----------------------------
PROMPT_V1 = """You are an information extraction system.

Extract ONE event record from the text and return ONLY a single valid JSON object.
The JSON MUST match the EventExtraction schema exactly. Do not output any extra text.

Rules:
- If a field is unknown, use null (or [] for lists).
- Provide evidence quotes: include 1 short quote per field if possible.
- If you are uncertain, add a short string to uncertainty_flags.
- Choose event_type ONLY from: protest, election, policy_change, violence, disaster, other.

Document ID: {doc_id}
Text: {text}
"""

PROMPT_V2 = """You are extracting ONE event record.

Return ONLY a single JSON object matching EventExtraction exactly (no extra text).

Event type definitions (pick ONE):
- protest: public demonstration, rally, march, strike
- election: voting, election officials, turnout, campaign event tied to an election
- policy_change: law/policy passed, amended, reversed, executive order/curfew rule
- violence: shooting, stabbing, armed clash, arrests due to violent incident
- disaster: earthquake, flood, landslide, storm causing damage/displacement
- other: none of the above clearly apply

Missingness:
- If unknown: null or [].
Evidence requirement:
- Put at least one quote supporting event_type in evidence.

Document ID: {doc_id}
Text: {text}
"""


# -----------------------------
# Mechanical review flags
# -----------------------------
def add_review_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["extraction_confidence"] = pd.to_numeric(out.get("extraction_confidence", 0.0), errors="coerce").fillna(0.0)

    out["flag_low_confidence"] = out["extraction_confidence"] < 0.70
    out["flag_missing_date"] = out["event_date_iso"].isna() | (out["event_date_iso"].astype(str).str.strip() == "")
    out["flag_missing_country"] = out["country"].isna() | (out["country"].astype(str).str.strip() == "")
    out["flag_geo_unknown"] = out["geo_precision"].isin(["unknown", "country_only"])

    # actors may be list or string depending on your pipeline
    if "actors" in out.columns:
        out["flag_actors_empty"] = out["actors"].apply(lambda x: (x is None) or (isinstance(x, list) and len(x) == 0) or (isinstance(x, str) and x.strip() == ""))
    elif "actors_json" in out.columns:
        out["flag_actors_empty"] = out["actors_json"].fillna("[]").apply(lambda s: len(json.loads(s)) == 0 if str(s).strip().startswith("[") else True)
    else:
        out["flag_actors_empty"] = True

    flag_cols = ["flag_low_confidence", "flag_missing_date", "flag_missing_country", "flag_geo_unknown", "flag_actors_empty"]
    out["needs_human_review"] = out[flag_cols].any(axis=1)

    return out


# -----------------------------
# Run extraction for one prompt
# -----------------------------
def run_extraction_with_prompt(docs_df: pd.DataFrame, prompt_template: str) -> pd.DataFrame:
    """
    Requires you already defined:
      - call_ollama_json(prompt) -> str
      - extract_first_json_object(raw) -> str
      - EventExtraction (Pydantic)
    """
    rows = []
    for i in range(len(docs_df)):
        doc_id = docs_df.loc[i, "doc_id"]
        text = docs_df.loc[i, "text"]

        prompt = prompt_template.format(doc_id=doc_id, text=text)

        parse_ok = True
        parse_error = ""
        raw = ""

        try:
            raw = call_ollama_json(prompt)
            json_str = extract_first_json_object(raw)
            record = json.loads(json_str)

            validated = EventExtraction(**record)
            out = validated.model_dump()

        except Exception as e:
            parse_ok = False
            parse_error = type(e).__name__

            # fallback record that still matches schema
            fallback = EventExtraction(
                doc_id=doc_id,
                event_type="other",
                event_date_iso=None,
                date_is_approximate=True,
                country=None,
                admin1_or_state=None,
                city_or_local=None,
                geo_precision="unknown",
                actors=[],
                outcome_summary=None,
                extraction_confidence=0.0,
                uncertainty_flags=[f"parse_or_validation_failed:{parse_error}"],
                evidence=[]
            )
            out = fallback.model_dump()

        out["raw_text"] = text
        out["model_raw_output"] = raw
        out["parse_ok"] = parse_ok
        out["parse_error"] = parse_error
        rows.append(out)

    df = pd.DataFrame(rows)
    df = add_review_flags(df)
    return df


# -----------------------------
# Evaluate event_type (gold set)
# -----------------------------
def evaluate_event_type(extractions_df: pd.DataFrame, gold_df: pd.DataFrame, label: str):
    eval_df = gold_df.merge(extractions_df[["doc_id", "event_type"]], on="doc_id", how="left")
    eval_df = eval_df.rename(columns={"event_type": "event_type_pred"})
    eval_df["event_type_pred"] = eval_df["event_type_pred"].fillna("other")

    # Classification report
    print("\n==============================")
    print(f"Classification report (event_type) — {label}")
    print("==============================")
    print(classification_report(
        eval_df["event_type_gold"],
        eval_df["event_type_pred"],
        labels=EVENT_TYPES,
        zero_division=0
    ))

    macro_f1 = f1_score(eval_df["event_type_gold"], eval_df["event_type_pred"], labels=EVENT_TYPES, average="macro", zero_division=0)
    acc = accuracy_score(eval_df["event_type_gold"], eval_df["event_type_pred"])

    # # flagged for human review across ALL docs
    n_flagged = int(extractions_df["needs_human_review"].sum())

    return {
        "prompt": label,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "n_flagged_for_review": n_flagged
    }


# -----------------------------
# MAIN: run both prompts + compare
# -----------------------------

extractions_v1 = run_extraction_with_prompt(docs_df, PROMPT_V1)
extractions_v2 = run_extraction_with_prompt(docs_df, PROMPT_V2)

# Save raw outputs for reproducibility
os.makedirs("outputs", exist_ok=True)
extractions_v1.to_csv("soda501_ps/07_llm_human/outputs/outputs/extractions_prompt_v1.csv", index=False)
extractions_v2.to_csv("soda501_ps/07_llm_human/outputsoutputs/extractions_prompt_v2.csv", index=False)

# Evaluate both
row1 = evaluate_event_type(extractions_v1, gold, "PROMPT_V1")
row2 = evaluate_event_type(extractions_v2, gold, "PROMPT_V2")

comparison = pd.DataFrame([row1, row2])
comparison = comparison[["prompt", "macro_f1", "accuracy", "n_flagged_for_review"]]

print("\n==============================")
print("Prompt comparison (required table)")
print("==============================")
print(comparison.to_string(index=False))

comparison.to_csv("outputs/prompt_comparison.csv", index=False)
print("\nWrote outputs/prompt_comparison.csv")
print("Also wrote outputs/extractions_prompt_v1.csv and outputs/extractions_prompt_v2.csv")