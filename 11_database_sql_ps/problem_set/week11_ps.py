

# WEEK 11: Database and SQL

### Conceptual Questions: ###

# A relational schema organizes data into structured tables (relations) with defined relationships,
# making it easier to store, manage, and analyze complex social data. A primary key uniquely identifies
# each row in a table (e.g., a candidate ID), while a foreign key links rows across tables
# (e.g., a contribution referencing a contributor ID), enabling efficient joins. 
# This design reduces duplication by storing each entity (candidates, contributors) once
# and referencing it rather than repeating information. In the (candidate, contributor, 
# contribution) example, contributions link to candidates and contributors via foreign keys,
#  allowing you to combine data across tables without redundancy.



### Applied Exercises: ###

# Q3:  Build the database + inspect the schema (synthetic data).

import sqlite3
import numpy as np
import pandas as pd
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# Step 1: Connect to SQLite database
# -----------------------------------------------------------------------------
con = sqlite3.connect("campaign_finance.db")
cur = con.cursor()

# -----------------------------------------------------------------------------
# Step 2: Drop existing tables 
# -----------------------------------------------------------------------------
cur.execute("DROP TABLE IF EXISTS contributions;")
cur.execute("DROP TABLE IF EXISTS contributors;")
cur.execute("DROP TABLE IF EXISTS candidates;")
con.commit()

# -----------------------------------------------------------------------------
# Step 3: Create tables
# -----------------------------------------------------------------------------
cur.execute("""
    CREATE TABLE candidates (
        candidate_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        party TEXT,
        office TEXT,
        winner INTEGER
    );
""")

cur.execute("""
    CREATE TABLE contributors (
        contributor_id INTEGER PRIMARY KEY,
        name TEXT,
        occupation TEXT,
        employer TEXT,
        state TEXT
    );
""")

cur.execute("""
    CREATE TABLE contributions (
        contribution_id INTEGER PRIMARY KEY,
        contributor_id INTEGER,
        candidate_id INTEGER,
        amount REAL,
        date TEXT,
        FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id),
        FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
    );
""")
con.commit()

# -----------------------------------------------------------------------------
# Step 4: Generate synthetic data
# -----------------------------------------------------------------------------
np.random.seed(12345)

# ---- candidates ----
candidate_ids = np.arange(1, 101)
candidates = pd.DataFrame({
    "candidate_id": candidate_ids,
    "name": [f"Candidate {i}" for i in candidate_ids],
    "party": np.random.choice(
        ["Democrat", "Republican", "Independent"],
        size=100,
        p=[0.45, 0.45, 0.10]
    ),
    "office": np.random.choice(
        ["Senate", "House", "Governor", "State Senate", "State House"],
        size=100
    ),
    "winner": np.random.choice([0, 1], size=100)
})

# ---- contributors ----
contributor_ids = np.arange(1, 100001)
state_abb = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

contributors = pd.DataFrame({
    "contributor_id": contributor_ids,
    "name": [f"Contributor {i}" for i in contributor_ids],
    "occupation": np.random.choice(
        ["Engineer", "Teacher", "Doctor", "Lawyer", "Business Owner"],
        size=100000
    ),
    "employer": [f"Company {i}" for i in np.random.randint(1, 5001, size=100000)],
    "state": np.random.choice(state_abb, size=100000)
})

# ---- contributions ----
contribution_ids = np.arange(1, 1000001)
start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
n_days = (end_date - start_date).days + 1
random_day_offsets = np.random.randint(0, n_days, size=1000000)

contributions = pd.DataFrame({
    "contribution_id": contribution_ids,
    "contributor_id": np.random.randint(1, 100001, size=1000000),
    "candidate_id": np.random.randint(1, 101, size=1000000),
    "amount": np.round(
        np.random.lognormal(mean=np.log(1000), sigma=1, size=1000000), 2
    ),
    "date": [(start_date + timedelta(days=int(d))).isoformat() for d in random_day_offsets]
})

# -----------------------------------------------------------------------------
# Step 5: Load data into SQLite
# -----------------------------------------------------------------------------
candidates.to_sql("candidates", con, if_exists="append", index=False, chunksize=5000)
contributors.to_sql("contributors", con, if_exists="append", index=False, chunksize=5000)
contributions.to_sql("contributions", con, if_exists="append", index=False, chunksize=5000)
con.commit()

# -----------------------------------------------------------------------------
# Step 6: Report row counts
# -----------------------------------------------------------------------------
print("\nRow counts:")
print(pd.read_sql_query("SELECT COUNT(*) AS n_candidates FROM candidates;", con)) # 100
print(pd.read_sql_query("SELECT COUNT(*) AS n_contributors FROM contributors;", con)) # 100000
print(pd.read_sql_query("SELECT COUNT(*) AS n_contributions FROM contributions;", con)) # 1000000

# -----------------------------------------------------------------------------
# Step 7: Show schema for each table
# -----------------------------------------------------------------------------
print("\nSchema for candidates:")
print(pd.read_sql_query("PRAGMA table_info(candidates);", con))

print("\nSchema for contributors:")
print(pd.read_sql_query("PRAGMA table_info(contributors);", con))

print("\nSchema for contributions:")
print(pd.read_sql_query("PRAGMA table_info(contributions);", con))


print("\nSample joined rows:")
sample_join = pd.read_sql_query("""
    SELECT
        co.contribution_id,
        co.amount,
        co.date,
        co.contributor_id,
        ctr.name AS contributor_name,
        co.candidate_id,
        ca.name AS candidate_name
    FROM contributions co
    JOIN contributors ctr
        ON co.contributor_id = ctr.contributor_id
    JOIN candidates ca
        ON co.candidate_id = ca.candidate_id
    LIMIT 5;
""", con)
print(sample_join)


# -----------------------------------------------------------------------------
#  Close connection
# -----------------------------------------------------------------------------
con.close()

# The contributor_id in the contributions table links each donation to a specific individual in the contributors table,
# while the candidate_id links that same donation to a candidate in the candidates table.


# Q4: Joins + aggregation 
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# connect to the existing SQLite database
# -----------------------------------------------------------------------------
con = sqlite3.connect("campaign_finance.db")

# -----------------------------------------------------------------------------
# SQL query: join + aggregation
# Required:
# - join contributions to candidates
# - restrict to amount > 1000
# - output party, total_amount, num_contributions
# -----------------------------------------------------------------------------
query_q4 = """
    SELECT
        ca.party AS party,
        SUM(co.amount) AS total_amount,
        COUNT(*) AS num_contributions
    FROM contributions co
    JOIN candidates ca
        ON co.candidate_id = ca.candidate_id
    WHERE co.amount > 1000
    GROUP BY ca.party
    ORDER BY total_amount DESC;
"""

party_summary = pd.read_sql_query(query_q4, con)

# -----------------------------------------------------------------------------
# Print clean output table
# -----------------------------------------------------------------------------
print("\nQ4: Total contributions by party for contributions > 1000")
print(party_summary)

# -----------------------------------------------------------------------------
# Round values for cleaner display
# -----------------------------------------------------------------------------
party_summary["total_amount"] = party_summary["total_amount"].round(2)

print("\nClean table:")
print(party_summary)

# -----------------------------------------------------------------------------
# Visualization: simple bar plot of total_amount by party
# -----------------------------------------------------------------------------

fig_dir = "/Users/songtao/Dropbox/26SP/SODA 501/soda501_ps/11_database_sql_ps/figure"
os.makedirs(fig_dir, exist_ok=True)

plt.figure()
plt.bar(party_summary["party"], party_summary["total_amount"])
plt.title("Total Contributions by Party (Amount > 1000)")
plt.xlabel("Party")
plt.ylabel("Total Amount")
plt.tight_layout()

output_path = os.path.join(fig_dir, "q4_total_contributions_by_party.png")

plt.savefig(output_path, dpi=150)
plt.show()


# -----------------------------------------------------------------------------
# Close database connection
# -----------------------------------------------------------------------------
con.close()



### Q5: Indexs + query plan ###

import sqlite3
import pandas as pd


con = sqlite3.connect("campaign_finance.db")

# -----------------------------------------------------------------------------
# Verify which indexes exist on contributions
# -----------------------------------------------------------------------------
query_indexes = """
    SELECT
        name,
        type,
        tbl_name,
        sql
    FROM sqlite_master
    WHERE type = 'index'
      AND tbl_name = 'contributions';
"""

indexes_df = pd.read_sql_query(query_indexes, con)

print("\nIndexes on contributions:")
print(indexes_df)

# -----------------------------------------------------------------------------
# Choose one query that filters by amount
# -----------------------------------------------------------------------------
test_query = """
    SELECT *
    FROM contributions
    WHERE amount > 5000;
"""

# -----------------------------------------------------------------------------
# Run EXPLAIN QUERY PLAN for that query
# -----------------------------------------------------------------------------
query_plan = pd.read_sql_query(f"EXPLAIN QUERY PLAN {test_query}", con)

print("\nEXPLAIN QUERY PLAN for query filtering by amount:")
print(query_plan)

# >>> print(query_plan)
#   id  parent  notused              detail
#   2       0      216  SCAN contributions

# -----------------------------------------------------------------------------
# Show a small sample of the filtered query result
# -----------------------------------------------------------------------------
sample_result = pd.read_sql_query("""
    SELECT *
    FROM contributions
    WHERE amount > 5000
    LIMIT 5;
""", con)

print("\nSample rows from filtered query:")
print(sample_result)

# -----------------------------------------------------------------------------
# Close connection
# -----------------------------------------------------------------------------
con.close()


## The query plan shows SCAN contributions, which means SQLite is performing a full table scan 
## and not using any index. This is inefficient because it must check all rows in the contributions table 
## to find those with amount > 5000. An index on the amount column (e.g., CREATE INDEX idx_contrib_amount ON contributions(amount);) 
## would help because the query filters directly on this variable.