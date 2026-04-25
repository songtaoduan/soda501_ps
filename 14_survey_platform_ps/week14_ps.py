

# Week 14. Survey Platform and Crowdsourcing

# Conceptual Questions

# MCAR means the probability of missingness is unrelated to any observed or unobserved variables (e.g., a random technical glitch),
# while MAR means missingness depends only on observed data. MNAR occurs when missingness depends on the unobserved value itself—for example,
# on survey platforms like Prolific or MTurk, respondents may skip sensitive questions (e.g., income or political views) specifically
# because of their true answers, or speed through and leave items blank to maximize payment per minute.
# This creates bias because the missing data are systematically different, meaning standard methods (like listwise deletion)
# can produce biased estimates and understate or misrepresent true relationships.

# Applied Exercises #
import os
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#os.chdir("/Users/songtao/Dropbox/26SP/SODA 501/soda501_ps/14_survey_platform_ps")
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------
# 3. Survey export: load + inspect
# ------------------------------------------------------------
survey_raw = pd.read_csv("data_raw/survey_export.csv")

print(survey_raw)
print(survey_raw.head())
print(survey_raw.dtypes)

print("Variables in export:")
print(list(survey_raw.columns))

print("""
Common platform variables include respondent ID, start time, end time,
duration, platform ID, IP address, consent, and attention-check responses.
One immediate cleaning issue is that missing values may appear as empty strings
or text such as 'Prefer not to say', so these should be recoded to NA.
""")


# ------------------------------------------------------------
# 4. Cleaning: names, types, missing values
# ------------------------------------------------------------
clean_cols = [
    re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", c)).strip("_").lower()
    for c in survey_raw.columns
]

survey_clean = survey_raw.copy()
survey_clean.columns = clean_cols

print(list(survey_clean.columns))

survey_clean["start_date"] = pd.to_datetime(
    survey_clean["startdate"], errors="coerce"
).dt.tz_localize(ZoneInfo("America/New_York"))

survey_clean["end_date"] = pd.to_datetime(
    survey_clean["enddate"], errors="coerce"
).dt.tz_localize(ZoneInfo("America/New_York"))

survey_clean = survey_clean.replace({
    "": np.nan,
    "Prefer not to say": np.nan,
    "prefer not to say": np.nan,
    "NA": np.nan,
    "N/A": np.nan
})

survey_clean["age_num"] = pd.to_numeric(survey_clean["age"], errors="coerce")
survey_clean["duration_seconds"] = pd.to_numeric(
    survey_clean["duration_seconds"], errors="coerce"
)

survey_clean["duration_min"] = survey_clean["duration_seconds"] / 60

print(survey_clean.head())
print(survey_clean.dtypes)

# ------------------------------------------------------------
# 5. Codebook: document data
# ------------------------------------------------------------
codebook = pd.DataFrame({
    "variable": [
        "responseid", "start_date", "end_date", "duration_seconds",
        "duration_min", "ipaddress", "platform", "prolific_id",
        "mturk_workerid", "consent", "age_num", "gender",
        "partyid", "attentioncheck", "q1_policy", "q2_policy",
        "q3_policy", "q4_policy"
    ],
    "description": [
        "Unique respondent identifier",
        "Survey start time",
        "Survey end time",
        "Survey duration in seconds",
        "Survey duration in minutes",
        "Respondent IP address",
        "Recruitment platform",
        "Prolific respondent ID",
        "MTurk worker ID",
        "Consent response",
        "Respondent age in years",
        "Respondent gender",
        "Party identification",
        "Attention-check response",
        "Policy attitude item 1",
        "Policy attitude item 2",
        "Policy attitude item 3",
        "Policy attitude item 4"
    ],
    "notes": [
        "Platform metadata",
        "Parsed from string",
        "Parsed from string",
        "Converted to numeric",
        "Created from duration_seconds",
        "Sensitive metadata",
        "Platform metadata",
        "Blank for non-Prolific respondents",
        "Blank for non-MTurk respondents",
        "Used for filtering",
        "Converted from string",
        "Cleaned missing strings",
        "Cleaned missing strings",
        "Used for quality check",
        "Survey item",
        "Survey item",
        "Survey item",
        "Survey item"
    ]
})


print(codebook)

codebook.to_csv("outputs/week_codebook.csv", index=False)

# ------------------------------------------------------------
# 6. Labeling: variable labels + value labels
# ------------------------------------------------------------

var_labels = {
    "responseid": "Unique respondent identifier",
    "duration_seconds": "Survey duration in seconds",
    "duration_min": "Survey duration in minutes",
    "age_num": "Age in years",
    "partyid": "Party identification",
    "attentioncheck": "Attention-check response"
}

survey_clean.attrs["var_labels"] = var_labels

party_map = {
    "Democrat": 1,
    "Independent": 2,
    "Republican": 3
}

survey_clean["party_id_num"] = survey_clean["partyid"].map(party_map)

value_labels_party = {
    1: "Democrat",
    2: "Independent",
    3: "Republican"
}

survey_clean.attrs["value_labels_party_id_num"] = value_labels_party

likert_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither": 3,
    "Agree": 4,
    "Strongly agree": 5
}

survey_clean["q1_num"] = survey_clean["q1_policy"].map(likert_map)
survey_clean["q2_num"] = survey_clean["q2_policy"].map(likert_map)
survey_clean["q3_num"] = survey_clean["q3_policy"].map(likert_map)
survey_clean["q4_num"] = survey_clean["q4_policy"].map(likert_map)

print(survey_clean[["responseid", "partyid", "party_id_num", "q1_policy", "q1_num"]])
print(survey_clean.attrs["var_labels"])
print(survey_clean.attrs["value_labels_party_id_num"])


# ------------------------------------------------------------
# 7. Quality checks: flags + summary
# ------------------------------------------------------------
survey_clean["flag_fast"] = survey_clean["duration_seconds"] < 120

survey_clean["flag_attention_fail"] = (
    survey_clean["attentioncheck"] != "Strongly disagree"
)

key_vars = survey_clean[
    ["age_num", "gender", "party_id_num", "q1_num", "q2_num", "q3_num", "q4_num"]
]

survey_clean["missing_share"] = key_vars.isna().mean(axis=1)

survey_clean["flag_missing_high"] = survey_clean["missing_share"] > 0.30

survey_clean["flag_straightline"] = (
    (survey_clean["q1_num"] == survey_clean["q2_num"]) &
    (survey_clean["q2_num"] == survey_clean["q3_num"]) &
    (survey_clean["q3_num"] == survey_clean["q4_num"])
)

survey_clean["flag_no_consent"] = survey_clean["consent"] != "Yes"

flag_summary = pd.DataFrame({
    "n_total": [len(survey_clean)],
    "n_fast": [survey_clean["flag_fast"].sum()],
    "n_attention_fail": [survey_clean["flag_attention_fail"].sum()],
    "n_missing_high": [survey_clean["flag_missing_high"].sum()],
    "n_straightline": [survey_clean["flag_straightline"].sum()],
    "n_no_consent": [survey_clean["flag_no_consent"].sum()]
})

print(flag_summary)

flagged_table = survey_clean[
    [
        "responseid", "platform", "duration_seconds", "attentioncheck",
        "missing_share", "flag_fast", "flag_attention_fail",
        "flag_missing_high", "flag_straightline", "flag_no_consent"
    ]
]

print(flagged_table)

# ------------------------------------------------------------
# 8. Analysis-ready dataset: filter + save + visualize
# ------------------------------------------------------------
survey_final = survey_clean.loc[
    (~survey_clean["flag_no_consent"]) &
    (~survey_clean["flag_fast"]) &
    (~survey_clean["flag_attention_fail"]) &
    (~survey_clean["flag_missing_high"])
].copy()

print("Rows before filtering:", len(survey_clean))
print("Rows after filtering:", len(survey_final))

survey_final.to_csv("data_processed/week_survey_clean.csv", index=False)

plt.figure()
plt.hist(survey_clean["duration_seconds"].dropna(), bins=12)
plt.title("Survey Duration Distribution")
plt.xlabel("Duration in seconds")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/duration_histogram.png", dpi=200)
plt.show()
plt.close()

plt.figure()
plt.hist(survey_clean["missing_share"].dropna(), bins=10)
plt.title("Missingness Share Across Key Variables")
plt.xlabel("Share missing")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/missingness_histogram.png", dpi=200)
plt.show()
plt.close()

platform_flags = (
    survey_clean.groupby("platform", dropna=False)
    .agg(
        n=("responseid", "size"),
        share_fast=("flag_fast", "mean"),
        share_attention_fail=("flag_attention_fail", "mean")
    )
    .reset_index()
)

print(platform_flags)

plt.figure()
plt.bar(platform_flags["platform"], platform_flags["share_fast"])
plt.title("Share of Speeders by Platform")
plt.xlabel("Platform")
plt.ylabel("Share flagged")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/speeders_by_platform.png", dpi=200)
plt.show()
plt.close()


