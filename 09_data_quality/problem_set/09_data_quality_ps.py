
### Conceptual questions: Placebo tests ##

# An outcome placebo replaces the true outcome with one that should not be affected by the treatment (e.g., a pre-treatment outcome or an unrelated variable). 
# If the pipeline is behaving well, the estimated effect should be **close to zero and statistically insignificant **. 
# A failure can occur if there is model misspecification or residual confounding, such as time trends or omitted variables
# that spuriously correlate treatment with the placebo outcome.

# A treatment permutation placebo randomly reshuffles the treatment assignment across units or time, 
# breaking any real causal link. A well-functioning pipeline should produce estimates centered around 
# zero across many permutations, with **no systematic significance**. This test can fail even when the true 
# effect is zero if there are violations of independence (e.g., clustering or spillovers) that the 
# permutation procedure does not properly account for.



## Applied Exercises ##

# 3. Measurement error simulation 
# Run "measurement_error_placebo_tests.py" Results are saved in outputs and figures folder.

#The first figure shows that the **naive estimate of the treatment effect (τ)** increases as measurement error (σᵤ) in the confounder grows. 
# When σᵤ is small, the naive estimate is close to the true value, but as σᵤ increases, the estimate becomes increasingly upward biased.
# This happens because measurement error weakens the ability of \(x_{obs}\) to control for the true confounder \(x_{true}\), 
# In contrast, the oracle estimate remains stable across all levels of σᵤ because
# it uses the true confounder, and the calibration approach partially corrects the bias.

# The second figure illustrates **attenuation bias in the confounder coefficient (β)**. 
# As σᵤ increases, the naive estimate of β shrinks toward zero 
# because classical measurement error reduces the signal-to-noise ratio in \(x_{obs}\),
# making it a weaker proxy for the true confounder. This leads to underestimation of the confounder’s effect.
# Meanwhile, the oracle estimate remains close to the true β, and the calibration approach recovers much of the lost signal.

# The key difference between the oracle and naive estimands is that the oracle model correctly conditions on the true confounder,
#  yielding unbiased estimates of both τ and β, while the naive model relies on a noisy proxy and therefore fails to fully adjust for confounding.
#  As a result, the naive estimand is biased—overstating the treatment effect and understating the confounder effect—especially as measurement error increases.



# 4. Validation subsample and regression calibration
# Validation subsample and regression calibration
# This script varies validation_share = [0.05, 0.20, 0.50]
# while holding sigma_u = 1.0 fixed, then reports
# tau_cal_mean and tau_naive_mean for each validation-share setting.

import os
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# -------------------------------------------------------------------
# Set working directory
# -------------------------------------------------------------------
os.chdir("/Users/songtao/Dropbox/26SP/SODA 501/soda501_ps/09_data_quality")

# Create folders if they do not exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
np.random.seed(123)

# -------------------------------------------------------------------
# Data generating process (same as tutorial)
# -------------------------------------------------------------------
n = 5000

# True confounder
x_true = np.random.normal(loc=0.0, scale=1.0, size=n)

# Treatment assignment correlated with x_true
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n)

# True outcome model
tau = 1.0
beta = 1.0
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n)
y = tau * d + beta * x_true + eps_y

df_base = pd.DataFrame({
    "y": y,
    "d": d,
    "x_true": x_true
})

# -------------------------------------------------------------------
# Settings for this exercise
# -------------------------------------------------------------------
sigma_u = 1.0
validation_share_grid = [0.05, 0.20, 0.50]
R = 30  # number of repetitions

# Storage
rows = []

print("\n--- Running validation-share simulation ---")

for validation_share in validation_share_grid:
    tau_naive_list = []
    tau_cal_list = []
    beta_naive_list = []
    beta_cal_list = []

    for r in range(R):
        # Draw measurement error
        u = np.random.normal(loc=0.0, scale=sigma_u, size=n)
        x_obs = x_true + u

        # Copy base data and add noisy confounder
        df = df_base.copy()
        df["x_obs"] = x_obs

        # Draw validation subsample for this repetition
        validation_idx = np.random.choice(
            np.arange(n),
            size=int(validation_share * n),
            replace=False
        )
        is_validation = np.zeros(n, dtype=bool)
        is_validation[validation_idx] = True

        # -----------------------------------------------------------
        # Naive regression: y ~ d + x_obs
        # -----------------------------------------------------------
        fit_naive = smf.ols("y ~ d + x_obs", data=df).fit()
        tau_naive_list.append(float(fit_naive.params["d"]))
        beta_naive_list.append(float(fit_naive.params["x_obs"]))

        # -----------------------------------------------------------
        # Regression calibration:
        # 1) On validation sample, estimate x_true ~ x_obs
        # 2) Predict x_hat for all observations
        # 3) Run y ~ d + x_hat
        # -----------------------------------------------------------
        df_val = df.loc[is_validation, ["x_true", "x_obs"]].copy()
        fit_cal = smf.ols("x_true ~ x_obs", data=df_val).fit()

        df["x_hat"] = fit_cal.predict(df[["x_obs"]])

        fit_calibrated = smf.ols("y ~ d + x_hat", data=df).fit()
        tau_cal_list.append(float(fit_calibrated.params["d"]))
        beta_cal_list.append(float(fit_calibrated.params["x_hat"]))

    rows.append({
        "validation_share": validation_share,
        "sigma_u": sigma_u,
        "tau_true": tau,
        "tau_naive_mean": float(np.mean(tau_naive_list)),
        "tau_naive_sd": float(np.std(tau_naive_list, ddof=1)),
        "tau_naive_q025": float(np.quantile(tau_naive_list, 0.025)),
        "tau_naive_q975": float(np.quantile(tau_naive_list, 0.975)),
        "tau_cal_mean": float(np.mean(tau_cal_list)),
        "tau_cal_sd": float(np.std(tau_cal_list, ddof=1)),
        "tau_cal_q025": float(np.quantile(tau_cal_list, 0.025)),
        "tau_cal_q975": float(np.quantile(tau_cal_list, 0.975)),
        "beta_naive_mean": float(np.mean(beta_naive_list)),
        "beta_cal_mean": float(np.mean(beta_cal_list))
    })

    print(f"done validation_share = {validation_share}")

# -------------------------------------------------------------------
# Results table
# -------------------------------------------------------------------
results_valshare = pd.DataFrame(rows).sort_values("validation_share").reset_index(drop=True)

print("\n--- Validation-share results ---")
print(results_valshare[[
    "validation_share",
    "sigma_u",
    "tau_true",
    "tau_naive_mean",
    "tau_cal_mean",
    "beta_naive_mean",
    "beta_cal_mean"
]])

# Save full results
results_valshare.to_csv("outputs/validation_share_calibration_results.csv", index=False)

# Save a clean table with only the main quantities requested
table_out = results_valshare[[
    "validation_share",
    "tau_naive_mean",
    "tau_cal_mean"
]].copy()

table_out["tau_naive_mean"] = table_out["tau_naive_mean"].round(4)
table_out["tau_cal_mean"] = table_out["tau_cal_mean"].round(4)

table_out.to_csv("outputs/validation_share_tau_table.csv", index=False)

print("\n--- Clean table for submission ---")
print(table_out)

# -------------------------------------------------------------------
# Plot: calibrated vs naive treatment estimate by validation share
# -------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(
    results_valshare["validation_share"],
    results_valshare["tau_naive_mean"],
    marker="o",
    label="Naive: y ~ d + x_obs"
)
plt.plot(
    results_valshare["validation_share"],
    results_valshare["tau_cal_mean"],
    marker="o",
    label="Calibration: y ~ d + x_hat"
)
plt.axhline(tau, linestyle="--", label="True tau")
plt.title(f"Treatment effect vs validation share (sigma_u={sigma_u})")
plt.xlabel("Validation share")
plt.ylabel("Estimated coefficient on d")
plt.legend()
plt.tight_layout()
plt.savefig("figures/validation_share_tau_comparison.png", dpi=200)
plt.close()

# -------------------------------------------------------------------
# Optional markdown table output
# -------------------------------------------------------------------
markdown_table = table_out.to_markdown(index=False)
with open("outputs/validation_share_tau_table.md", "w") as f:
    f.write(markdown_table)

print("\nDone. Files written:")
print("  outputs/validation_share_calibration_results.csv")
print("  outputs/validation_share_tau_table.csv")
print("  outputs/validation_share_tau_table.md")
print("  figures/validation_share_tau_comparison.png")

# As the validation share increases, the calibrated estimate should generally move closer to the true treatment effect and become more stable across repetitions. 
# Calibration helps because the validation subsample contains both \(x_{true}\) and \(x_{obs}\), allowing us to estimate the relationship between the noisy proxy and the true confounder, 
# then use that relationship to recover a better-adjusted control variable. In contrast, the naive model uses \(x_{obs}\) directly, so it leaves more residual confounding when measurement error is substantial.
# A larger validation sample improves the precision of the calibration step, which usually makes \( \tau_{cal} \) less noisy and closer to the oracle benchmark.

# However, calibration relies on assumptions such as a correctly specified calibration model, a validation subsample that is representative of the full sample,
#  and measurement error that behaves similarly across groups. If these assumptions fail, calibration may still be biased. 
# In real social data, one reason calibration may fail is that measurement error is often nonclassical—for example, 
# survey misreporting may depend on treatment status, education, or political preferences. In that case, the simple 
# linear relationship estimated in the validation sample may not fully recover the true confounder, so the calibrated estimate can remain biased.


# Placebo tests: outcome placebo and treatment permutation

### Outcome Placebo ###

# The estimated coefficient on \(d\) in the outcome placebo regression is close to zero (approximately 0 when \(\sigma_u = 0\),
#  but increases slightly as measurement error grows). This is expected because the placebo outcome \(y_{placebo}\) is 
# constructed to have no causal relationship with the treatment. Therefore, a well-functioning pipeline should not detect any systematic effect of \(d\) on this outcome. 
# Small deviations from zero at higher levels of measurement error reflect residual confounding due to imperfect control of the noisy covariate.

### Treatment Permutation Placebo ###

# The observed naive estimate is \( \hat{\tau}_{obs} \approx 1.452 \), and the empirical two-sided p-value is approximately 0.002. The permutation histogram shows that the distribution of coefficients under random reassignment of treatment is tightly centered around zero, representing the null hypothesis of no treatment effect. Under this null, any association between treatment and outcome is purely due to chance.

# The observed estimate lies far in the tail of this permutation distribution, indicating that it is extremely unlikely to arise under the null hypothesis. This provides strong evidence against the null and suggests that the estimated treatment effect is not driven by random variation. In other words, the treatment effect detected by the model is substantively large relative to what would be expected under random assignment.

# Placebo tests like this help diagnose pipeline issues because they simulate a world where the treatment has no real effect. If the pipeline were flawed (e.g., due to data leakage, coding errors, or overfitting), we might observe large or significant effects even in the permuted data. The fact that the permutation distribution is centered near zero suggests that the pipeline is behaving correctly in this case.