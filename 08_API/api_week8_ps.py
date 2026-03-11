###############################################################################
# API Use + Forecasting Tutorial: Python
# Week 9 Problem Set
###############################################################################

### Conceptual Questions ###

# 1. Collecting data via an API offers two key advantages in social science research. 
# First, APIs provide structured and standardized data (e.g., JSON or XML), which reduces
# parsing errors and improves reliability compared to web scraping. 
# Second, APIs are typically supported by official documentation, making data 
# collection more transparent and consistent across researchers. 
# However, APIs also have limitations: they often impose rate limits or access restrictions 
# that constrain data collection, and endpoints or versions may change over time, 
# potentially breaking replication scripts or altering available variables. 
# To document API-based data provenance, I would record the API name, endpoint URLs, 
# query parameters, date of access, API version, authentication method, and provide 
# the full data collection code and raw responses so another researcher could reproduce the dataset later


### Applied Exercises ###

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install pandas numpy matplotlib statsmodels fredapi pyreadr plotly lxml requests

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
import statsmodels.formula.api as smf

# FRED API wrapper
from fredapi import Fred

# For reading .rds (RDS) files in Python (state-level poll/census data)
import pyreadr

# For a quick US states choropleth
import plotly.express as px

import os

# -----------------------------------------------------------------------------
# Part 1: Presidential vote data (national-level)
# -----------------------------------------------------------------------------
# Read in the presidential election vote data
vote_data = pd.read_csv("soda501_ps/08_API/1976-2020-president.csv")

# Keep only Democrat and Republican votes
vote_data = vote_data[
    vote_data["party_detailed"].isin(["DEMOCRAT", "REPUBLICAN"])
].copy()

# Summarize votes by year, candidate, party (mimics ddply summarize in R)
vote_data = (
    vote_data
    .groupby(["year", "candidate", "party_detailed"], as_index=False)
    .agg(
        candidatevotes=("candidatevotes", "sum"),
        totalvotes=("totalvotes", "sum")
    )
)

# Drop OTHER and blank candidate entries (mimics R filters)
vote_data = vote_data[
    (~vote_data["candidate"].isin(["OTHER", ""])) &
    (vote_data["candidate"].notna())
].copy()

# Compute vote percent
vote_data["vote_pct"] = vote_data["candidatevotes"] / vote_data["totalvotes"]

# Election years used in this dataset
election_years = np.sort(vote_data["year"].unique())


# -----------------------------------------------------------------------------
# Part 2: Pulling economic indicators from FRED (Q1/Q2 of election years)
# -----------------------------------------------------------------------------
# NOTE: Replace with your own key (students should get one from FRED).
fred_api_key = "81bb4fa2db1d8490b5316fb0814bf61c"
fred = Fred(api_key=fred_api_key)

# Define observation window based on the election years in the vote data
obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

# --- Unemployment (UNRATE) ---
# FRED returns a time series with dates; we convert to quarterly and keep Q1/Q2
unrate = fred.get_series("UNRATE", observation_start=obs_start, observation_end=obs_end)
unrate = unrate.to_frame(name="unemployment_rate")
unrate.index = pd.to_datetime(unrate.index)
unrate = unrate.resample("Q").mean().reset_index().rename(columns={"index": "date"})
unrate["year"] = unrate["date"].dt.year
unrate["quarter"] = unrate["date"].dt.quarter
unemployment_data = unrate[
    (unrate["year"].isin(election_years)) &
    (unrate["quarter"] <= 2)
][["year", "quarter", "unemployment_rate"]].copy()

# --- GDP (GDP) ---
gdp = fred.get_series("GDP", observation_start=obs_start, observation_end=obs_end)
gdp = gdp.to_frame(name="gdp")
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.resample("Q").mean().reset_index().rename(columns={"index": "date"})
gdp["year"] = gdp["date"].dt.year
gdp["quarter"] = gdp["date"].dt.quarter
gdp_data = gdp[
    (gdp["year"].isin(election_years)) &
    (gdp["quarter"] <= 2)
][["year", "quarter", "gdp"]].copy()

# --- CPI (CPIAUCSL) ---
cpi = fred.get_series("CPIAUCSL", observation_start=obs_start, observation_end=obs_end)
cpi = cpi.to_frame(name="cpi")
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi.resample("Q").mean().reset_index().rename(columns={"index": "date"})
cpi["year"] = cpi["date"].dt.year
cpi["quarter"] = cpi["date"].dt.quarter
cpi_data = cpi[
    (cpi["year"].isin(election_years)) &
    (cpi["quarter"] <= 2)
][["year", "quarter", "cpi"]].copy()

# (Optional, for teaching) inflation rate example (year-over-year using Q1 vs Q3 lag etc.)
# The original R code computed inflation_rate and then dropped it before widening.
# We replicate the same idea but do not use it in the final wide dataset.
inflation_data = cpi_data.sort_values(["year", "quarter"]).copy()
inflation_data["inflation_rate"] = (
    (inflation_data["cpi"] / inflation_data["cpi"].shift(2) - 1) * 100
)

# Combine all economic data into one long table keyed by (year, quarter)
combined_long = (
    unemployment_data
    .merge(gdp_data, on=["year", "quarter"], how="outer")
    .merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer")
    .sort_values(["year", "quarter"])
)

# Pivot wider like R pivot_wider(names_from=quarter, values_from=c(...), names_sep="_Q")
combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi"],
    aggfunc="first"
)

# Flatten column names to match the R naming style, e.g. unemployment_rate_Q1
combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()


# -----------------------------------------------------------------------------
# Part 3: Merge vote data + economic data and build national forecast features
# -----------------------------------------------------------------------------
forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()

# Incumbent indicator (hard-coded, sequential assignments like the R mutate/ifelse chain)
forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD") & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY") & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD") & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL") & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.") & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.") & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.") & (forecast_data["year"] == 2020), "incumbent"] = 1

# Quarter-to-quarter changes (Q2 - Q1), matching the R code
forecast_data["gdp_change"] = forecast_data["gdp_Q2"] - forecast_data["gdp_Q1"]
forecast_data["cpi_change"] = forecast_data["cpi_Q2"] - forecast_data["cpi_Q1"]
forecast_data["unemploy_change"] = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"]

# Split training (pre-2020) vs testing (2020) (prevent data leakage)
forecast_data_training = forecast_data[forecast_data["year"] < 2020].copy()
forecast_data_testing  = forecast_data[forecast_data["year"] == 2020].copy()

# Fit the national OLS model
# R: vote_pct ~ incumbent * unemploy_change + party_detailed + poly(year, 2, raw = T)
# Python: use year + year^2 explicitly
train_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=forecast_data_training
).fit()

# Generate predictions for training data
forecast_data_training["pred_vote"] = train_ols.predict(forecast_data_training)
print(forecast_data_training[["vote_pct", "pred_vote"]].head(20))

# Generate predictions for test data (2020)
test_pred = train_ols.predict(forecast_data_testing)
print("\n2020 test predictions (first few):")
print(test_pred.head())


# -----------------------------------------------------------------------------
# Part 4: State-level model (poll + census + economy)
# -----------------------------------------------------------------------------
# Load pre-existing poll and census data (RDS) and convert to pandas DataFrame
# NOTE: Update the path to wherever the RDS file lives on your system.
poll_census_path = "soda501_ps/08_API/poll_census_data.rds"
poll_census_obj = pyreadr.read_r(poll_census_path)
poll_census_data = list(poll_census_obj.values())[0]

# Prepare economic data for merging with state-level data (distinct year-level fields)
forecast_econ = forecast_data[
    ["year",
     "unemployment_rate_Q1", "unemployment_rate_Q2",
     "gdp_Q1", "gdp_Q2",
     "cpi_Q1", "cpi_Q2",
     "gdp_change", "cpi_change", "unemploy_change"]
].drop_duplicates()

# Merge state-level poll/census data with economic data
state_data = poll_census_data.merge(forecast_econ, on="year", how="left")

# Fit the state-level OLS model (training: year < 2020)
# R: vote_pct ~ poll_avg + year + party_simplified + white + black + asian + hispanic
pred_results = smf.ols(
    "vote_pct ~ poll_avg + year + C(party_simplified) + white + black + asian + hispanic",
    data=state_data[state_data["year"] < 2020]
).fit()

# Out-of-sample predictions for 2020 and beyond
out_of_sample = pred_results.predict(state_data[state_data["year"] >= 2020])

# Prepare election outcomes table (actual + predicted)
elect_outcomes = state_data[state_data["year"] >= 2020][
    ["year", "state_po", "party_simplified", "candidate", "vote_pct"]
].copy()

elect_outcomes["vote_pred"] = out_of_sample.values


# -----------------------------------------------------------------------------
# Part 5: 2020 vote difference (Biden minus Trump) and a map
# -----------------------------------------------------------------------------
# Create a 2020-only dataset
elect_2020 = elect_outcomes[elect_outcomes["year"] == 2020].copy()

# Standardize candidate names into a simple label for pivoting
elect_2020["candidate_simple"] = elect_2020["candidate"].astype(str).str.lower()
elect_2020.loc[elect_2020["candidate_simple"].str.contains("biden"), "candidate_simple"] = "biden"
elect_2020.loc[elect_2020["candidate_simple"].str.contains("trump"), "candidate_simple"] = "trump"

# Pivot wide like R pivot_wider(... names_glue = "{candidate}_{.value}")
wide_2020 = elect_2020.pivot_table(
    index=["state_po", "year"],
    columns="candidate_simple",
    values=["vote_pct", "vote_pred"],
    aggfunc="first"
)

# Flatten column names to match the R naming style (candidate_value)
wide_2020.columns = [f"{cand}_{val}" for val, cand in wide_2020.columns]
wide_2020 = wide_2020.reset_index()

# Vote difference (Biden minus Trump), matching the R intent
vote_diff_2020 = wide_2020.copy()
vote_diff_2020["vote_diff"] = vote_diff_2020["biden_vote_pct"] - vote_diff_2020["trump_vote_pct"]
vote_diff_2020 = vote_diff_2020[["state_po", "vote_diff"]].drop_duplicates()

# (Optional) Remove AK and HI to mimic the R map example
vote_diff_2020 = vote_diff_2020[~vote_diff_2020["state_po"].isin(["AK", "HI"])].copy()

# Plot a simple choropleth map of the vote difference

output_dir = "/Users/songtao/Dropbox/26SP/SODA 501/soda501_ps/08_API/output"
os.makedirs(output_dir, exist_ok=True)


fig = px.choropleth(
    vote_diff_2020,
    locations="state_po",
    locationmode="USA-states",
    color="vote_diff",
    color_continuous_midpoint=0,
    scope="usa",
    title="2020 Vote Share Difference (Biden − Trump)"
)

fig.write_html(f"{output_dir}/vote_diff_2020_map.html")

fig.show()

# -----------------------------------------------------------------------------
# Q5: Build a better out-of-sample forecaster (hold out 2020)
# Improvements:
#   (1) add extra FRED indicators: PAYEMS, INDPRO
#   (2) switch model family from OLS to ridge regression
# -----------------------------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Helper function to pull a FRED series and keep Q1/Q2 ----------
def get_fred_q1q2(series_id, value_name):
    s = fred.get_series(series_id, observation_start=obs_start, observation_end=obs_end)
    s = s.to_frame(name=value_name)
    s.index = pd.to_datetime(s.index)
    s = s.resample("QE").mean().reset_index().rename(columns={"index": "date"})
    s["year"] = s["date"].dt.year
    s["quarter"] = s["date"].dt.quarter
    s = s[
        (s["year"].isin(election_years)) &
        (s["quarter"] <= 2)
    ][["year", "quarter", value_name]].copy()
    return s

# ---------- Additional FRED indicators ----------
# Total nonfarm payrolls
payems_data = get_fred_q1q2("PAYEMS", "payems")

# Industrial production index
indpro_data = get_fred_q1q2("INDPRO", "indpro")

# Merge into long economic table
combined_long_plus = (
    combined_long
    .merge(payems_data, on=["year", "quarter"], how="left")
    .merge(indpro_data, on=["year", "quarter"], how="left")
    .sort_values(["year", "quarter"])
)

# Pivot wide
combined_wide_plus = combined_long_plus.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi", "payems", "indpro"],
    aggfunc="first"
)

combined_wide_plus.columns = [f"{var}_Q{q}" for var, q in combined_wide_plus.columns]
combined_wide_plus = combined_wide_plus.reset_index()

# Merge back to vote data
forecast_data_plus = vote_data.merge(combined_wide_plus, on="year", how="left").copy()

# Recreate incumbent variable
forecast_data_plus["incumbent"] = 0
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "FORD, GERALD") & (forecast_data_plus["year"] == 1976), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "CARTER, JIMMY") & (forecast_data_plus["year"] == 1980), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "REAGAN, RONALD") & (forecast_data_plus["year"] == 1984), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data_plus["year"] == 1992), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "CLINTON, BILL") & (forecast_data_plus["year"] == 1996), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "BUSH, GEORGE W.") & (forecast_data_plus["year"] == 2004), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "OBAMA, BARACK H.") & (forecast_data_plus["year"] == 2012), "incumbent"] = 1
forecast_data_plus.loc[(forecast_data_plus["candidate"] == "TRUMP, DONALD J.") & (forecast_data_plus["year"] == 2020), "incumbent"] = 1

# Feature engineering: quarter-to-quarter changes
forecast_data_plus["gdp_change"] = forecast_data_plus["gdp_Q2"] - forecast_data_plus["gdp_Q1"]
forecast_data_plus["cpi_change"] = forecast_data_plus["cpi_Q2"] - forecast_data_plus["cpi_Q1"]
forecast_data_plus["unemploy_change"] = forecast_data_plus["unemployment_rate_Q2"] - forecast_data_plus["unemployment_rate_Q1"]
forecast_data_plus["payems_change"] = forecast_data_plus["payems_Q2"] - forecast_data_plus["payems_Q1"]
forecast_data_plus["indpro_change"] = forecast_data_plus["indpro_Q2"] - forecast_data_plus["indpro_Q1"]

# A simple nonlinear term (another improvement in functional form)
forecast_data_plus["unemploy_change_sq"] = forecast_data_plus["unemploy_change"] ** 2

# Train/test split
train_df = forecast_data_plus[forecast_data_plus["year"] < 2020].copy()
test_df  = forecast_data_plus[forecast_data_plus["year"] == 2020].copy()

# ---------- Baseline OLS  ----------
baseline_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=train_df
).fit()

test_df["pred_baseline"] = baseline_ols.predict(test_df)

# ---------- Improved model: ridge with extra indicators ----------
numeric_features = [
    "incumbent",
    "year",
    "unemploy_change",
    "unemploy_change_sq",
    "gdp_change",
    "cpi_change",
    "payems_change",
    "indpro_change"
]

categorical_features = ["party_detailed"]

X_train = train_df[numeric_features + categorical_features]
y_train = train_df["vote_pct"]
X_test  = test_df[numeric_features + categorical_features]
y_test  = test_df["vote_pct"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

ridge_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RidgeCV(alphas=np.logspace(-3, 3, 50)))
])

ridge_model.fit(X_train, y_train)
test_df["pred_improved"] = ridge_model.predict(X_test)

# ---------- Evaluate out-of-sample performance ----------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

results = pd.DataFrame({
    "model": ["Baseline OLS", "Improved Ridge"],
    "MAE": [
        mean_absolute_error(y_test, test_df["pred_baseline"]),
        mean_absolute_error(y_test, test_df["pred_improved"])
    ],
    "RMSE": [
        rmse(y_test, test_df["pred_baseline"]),
        rmse(y_test, test_df["pred_improved"])
    ]
})

print("\nOut-of-sample results (test year = 2020):")
print(results)

print("\n2020 actual vs predicted:")
print(
    test_df[["year", "candidate", "party_detailed", "vote_pct", "pred_baseline", "pred_improved"]]
    .sort_values("party_detailed")
)

# Save results

results.to_csv(f"{output_dir}/oos_model_comparison_2020.csv", index=False)
test_df[["year", "candidate", "party_detailed", "vote_pct", "pred_baseline", "pred_improved"]].to_csv(
    f"{output_dir}/oos_predictions_2020.csv", index=False
)



# Q6 

# Reshape for plotting
plot_df = test_df[[
    "candidate",
    "party_detailed",
    "vote_pct",
    "pred_baseline",
    "pred_improved"
]].copy()

plot_long = plot_df.melt(
    id_vars=["candidate", "party_detailed", "vote_pct"],
    value_vars=["pred_baseline", "pred_improved"],
    var_name="model",
    value_name="predicted_vote"
)

# Rename labels for clarity
plot_long["model"] = plot_long["model"].replace({
    "pred_baseline": "Baseline OLS",
    "pred_improved": "Improved Ridge"
})

# Predicted vs actual scatter
fig = px.scatter(
    plot_long,
    x="vote_pct",
    y="predicted_vote",
    color="model",
    hover_data=["candidate", "party_detailed"],
    title="Out-of-Sample Model Fit (2020 Election)",
    labels={
        "vote_pct": "Actual Vote Share (%)",
        "predicted_vote": "Predicted Vote Share (%)"
    }
)

# Add 45-degree perfect prediction line
fig.add_shape(
    type="line",
    x0=plot_long["vote_pct"].min(),
    y0=plot_long["vote_pct"].min(),
    x1=plot_long["vote_pct"].max(),
    y1=plot_long["vote_pct"].max(),
    line=dict(dash="dash")
)

fig.write_html(f"{output_dir}/model_fit_predicted_vs_actual_2020.html")

fig.show()