
# Week 13. Record Linkage Problem Set

####Conceptual Questions ###

# Deterministic matching can fail because it requires exact agreement across fields, 
# so even minor errors like typos or small differences prevent true matches from being identified. 
# This creates a trade-off between false matches (false positives) and missed matches (false negatives): 
# strict rules reduce false matches but increase missed matches. Probabilistic matching (e.g., fastLink)
# manages this trade-off by assigning match probabilities and allowing partial disagreement,
# so researchers can choose a threshold that balances accuracy and coverage.# 

#### Applied Exercises ###
install.packages(c("fastLink", "dplyr", "ggplot2", "stringdist"))

library(fastLink)
library(dplyr)
library(ggplot2)
library(stringdist)

# -----------------------------------------------------------------------------
# 1. Generate the synthetic datasets
# -----------------------------------------------------------------------------

set.seed(123)
n <- 10000

df_a <- data.frame(
  id = 1:n,
  firstname = sample(
    c("John", "Jane", "Michael", "Emily", "David", "Sarah",
      "William", "Emma", "James", "Olivia"),
    n, replace = TRUE
  ),
  lastname = sample(
    c("Smith", "Johnson", "Williams", "Brown", "Jones",
      "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"),
    n, replace = TRUE
  ),
  birthyear = sample(1970:2000, n, replace = TRUE),
  zipcode = sample(10000:20000, n, replace = TRUE)
) %>%
  distinct()

df_b <- df_a

mod_firstname <- runif(nrow(df_b)) < 0.25
mod_lastname  <- runif(nrow(df_b)) < 0.25
mod_birthyear <- runif(nrow(df_b)) < 0.25

idx_firstname <- which(mod_firstname)
for (i in idx_firstname) {
  firstname <- df_b$firstname[i]
  chars <- strsplit(firstname, "")[[1]]
  num_replace <- sample(1:length(chars), 1)
  positions <- sample(1:length(chars), num_replace)
  for (pos in positions) {
    chars[pos] <- sample(letters, 1)
  }
  df_b$firstname[i] <- paste0(chars, collapse = "")
}

idx_lastname <- which(mod_lastname)
for (i in idx_lastname) {
  lastname <- df_b$lastname[i]
  chars <- strsplit(lastname, "")[[1]]
  num_replace <- sample(1:length(chars), 1)
  positions <- sample(1:length(chars), num_replace)
  for (pos in positions) {
    chars[pos] <- sample(letters, 1)
  }
  df_b$lastname[i] <- paste0(chars, collapse = "")
}

idx_birthyear <- which(mod_birthyear)
birthyear_shift <- sample(-2:2, length(idx_birthyear), replace = TRUE)
df_b$birthyear[idx_birthyear] <- df_b$birthyear[idx_birthyear] + birthyear_shift

write.csv(df_a, "dataset_a.csv", row.names = FALSE)
write.csv(df_b, "dataset_b.csv", row.names = FALSE)


df_a <- read.csv("dataset_a.csv")
df_b <- read.csv("dataset_b.csv")


summary(df_a)
summary(df_b)
head(df_a)
head(df_b)

# -----------------------------------------------------------------------------
#  Deterministic (exact) matching
# -----------------------------------------------------------------------------
det_matches <- merge(
  df_a, df_b,
  by = c("firstname", "lastname", "birthyear", "zipcode"),
  suffixes = c(".a", ".b")
)

det_n <- nrow(det_matches)
det_rate <- det_n / nrow(df_a)

cat("Number of deterministic matches:", det_n, "\n")
cat("Deterministic match rate:", round(det_rate, 4), "\n")

# save deterministic matches
write.csv(det_matches, "deterministic_matches.csv", row.names = FALSE)

# The deterministic match count is substantially below the total number of true linked records because exact matching requires 
# firstname, lastname, birthyear, and zipcode to agree perfectly. In this simulation, dataset_b was intentionally corrupted with 
# typos in first names, typos in last names, and small shifts in birth year for many records. Even if two rows refer to the same 
# underlying person, a one-character typo or a one-year difference is enough to break an exact match. 
# As a result, deterministic matching is very strict and tends to miss many true matches when the data contain noise. 

# -----------------------------------------------------------------------------
#  Probabilistic matching with fastLink + threshold curve
# -----------------------------------------------------------------------------

fl_out <- fastLink(
  dfA = df_a,
  dfB = df_b,
  varnames = c("firstname", "lastname", "birthyear", "zipcode"),
  return.all = TRUE
)

threshold_grid <- seq(0, 1, by = 0.01)

match_counts <- c()

for (t in threshold_grid) {
  matches_t <- getMatches(
    dfA = df_a,
    dfB = df_b,
    fl.out = fl_out,
    threshold.match = t
  )
  match_counts <- c(match_counts, nrow(matches_t))
}

threshold_results <- data.frame(
  threshold = threshold_grid,
  matches = match_counts
)

print(head(threshold_results, 15))
print(tail(threshold_results, 15))

# Plot: number of matches vs threshold
p_threshold <- ggplot(threshold_results, aes(x = threshold, y = matches)) +
  geom_line(linewidth = 1) +
  labs(
    title = "Number of Matches vs. Posterior Threshold",
    x = "Threshold",
    y = "Number of Matches"
  ) +
  theme_bw()

print(p_threshold)

ggsave(
  filename = "matches_vs_threshold.png",
  plot = p_threshold,
  width = 8,
  height = 5,
  dpi = 300
)

# "The threshold curve should generally slope downward as the threshold increases.",
#  "At low thresholds, fastLink accepts many candidate pairs, including weaker and more uncertain matches,",
#  "so the total match count is large. As the threshold increases, only pairs with stronger posterior evidence",
#  "are retained, so the number of matches falls. This happens because a higher cutoff is more conservative:",
#  it improves match certainty but excludes pairs with partial disagreement caused by typos or birth-year noise."


# -----------------------------------------------------------------------------
# Match quality, threshold choice, and posterior interpretation
# -----------------------------------------------------------------------------
# Create a very low-threshold candidate set
matches_low <- getMatches(
  dfA = df_a,
  dfB = df_b,
  fl.out = fl_out,
  threshold.match = 0.000001
)

# Check columns to understand returned structure
names(matches_low)


# use id to align true pairs because df_b was created from df_a.
comp_data_by_match <- data.frame()

for (upper in seq(0.1, 1, by = 0.1)) {

  temp_data <- matches_low %>%
    filter(posterior > upper - 0.1, posterior <= upper)

  if (nrow(temp_data) == 0) next

  df_a_temp <- df_a %>% filter(id %in% temp_data$id)
  df_b_temp <- df_b %>% filter(id %in% temp_data$id)

  compare_data <- df_a_temp %>%
    inner_join(df_b_temp, by = "id", suffix = c(".a", ".b"))

  if (nrow(compare_data) == 0) next

  first_dist <- stringdist(
    compare_data$firstname.a,
    compare_data$firstname.b,
    method = "lv"
  )

  last_dist <- stringdist(
    compare_data$lastname.a,
    compare_data$lastname.b,
    method = "lv"
  )

  birth_diff <- abs(compare_data$birthyear.a - compare_data$birthyear.b)

  comp_data_by_match <- bind_rows(
    comp_data_by_match,
    data.frame(
      posterior_bin = paste0(round(upper - 0.1, 1), "-", round(upper, 1)),
      first_name_distance = first_dist,
      last_name_distance = last_dist,
      birth_year_distance = birth_diff
    )
  )
}

# Summarize by posterior bin
quality_summary <- comp_data_by_match %>%
  group_by(posterior_bin) %>%
  summarize(
    mean_first_name_distance = mean(first_name_distance),
    median_first_name_distance = median(first_name_distance),
    mean_last_name_distance = mean(last_name_distance),
    median_last_name_distance = median(last_name_distance),
    mean_birth_year_distance = mean(birth_year_distance),
    median_birth_year_distance = median(birth_year_distance),
    n = n(),
    .groups = "drop"
  )

print(quality_summary)

write.csv(quality_summary, "posterior_bin_quality_summary.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
#  Boxplots by posterior bin
# -----------------------------------------------------------------------------
# First-name distance
p_first <- ggplot(comp_data_by_match, aes(x = posterior_bin, y = first_name_distance)) +
  geom_boxplot() +
  labs(
    title = "First-Name Levenshtein Distance by Posterior Bin",
    x = "Posterior Bin",
    y = "Levenshtein Distance"
  ) +
  theme_bw()

print(p_first)

ggsave(
  filename = "first_name_distance_by_posterior_bin.png",
  plot = p_first,
  width = 8,
  height = 5,
  dpi = 300
)

# Last-name distance
p_last <- ggplot(comp_data_by_match, aes(x = posterior_bin, y = last_name_distance)) +
  geom_boxplot() +
  labs(
    title = "Last-Name Levenshtein Distance by Posterior Bin",
    x = "Posterior Bin",
    y = "Levenshtein Distance"
  ) +
  theme_bw()

print(p_last)

ggsave(
  filename = "last_name_distance_by_posterior_bin.png",
  plot = p_last,
  width = 8,
  height = 5,
  dpi = 300
)

# Birth-year distance
p_birth <- ggplot(comp_data_by_match, aes(x = posterior_bin, y = birth_year_distance)) +
  geom_boxplot() +
  labs(
    title = "Birth-Year Difference by Posterior Bin",
    x = "Posterior Bin",
    y = "Absolute Difference in Birth Year"
  ) +
  theme_bw()

print(p_birth)

ggsave(
  filename = "birth_year_distance_by_posterior_bin.png",
  plot = p_birth,
  width = 8,
  height = 5,
  dpi = 300
)

# -----------------------------------------------------------------------------
# Choose a final threshold
# -----------------------------------------------------------------------------

candidate_thresholds <- c(0.50, 0.70, 0.80, 0.90, 0.95)

threshold_choice_table <- data.frame()

for (t in candidate_thresholds) {
  matches_t <- getMatches(
    dfA = df_a,
    dfB = df_b,
    fl.out = fl_out,
    threshold.match = t
  )

  threshold_choice_table <- bind_rows(
    threshold_choice_table,
    data.frame(
      threshold = t,
      matches = nrow(matches_t)
    )
  )
}

print(threshold_choice_table)

# Choose one threshold manually after looking at:
# 1. the threshold curve
# 2. the posterior-bin distance plots

final_threshold <- 0.85

final_matches <- getMatches(
  dfA = df_a,
  dfB = df_b,
  fl.out = fl_out,
  threshold.match = final_threshold
)

cat("Final chosen threshold:", final_threshold, "\n")
cat("Number of final probabilistic matches:", nrow(final_matches), "\n")

write.csv(final_matches, "final_probabilistic_matches.csv", row.names = FALSE)

# The posterior scores can be interpreted as the model's estimated probability that a record pair is a true match. 
# Higher-posterior bins should show smaller first-name distances, smaller last-name distances, and smaller birth-year
# differences, which indicates better match quality. A reasonable threshold is one near the point where
# the threshold curve begins to drop more sharply but the retained matches still have strong quality diagnostics. 
# For example, choosing a threshold around 0.80 to 0.90 often provides a good balance between recall and precision 
# in this simulation, because it keeps many likely true matches while filtering out weaker pairs. 


# Deterministic matching yields far fewer matches because it requires exact agreement across all fields,
# making it highly sensitive to typos and small data errors. In contrast, probabilistic matching identifies 
# many more matches by allowing partial disagreement and assigning posterior probabilities to candidate pairs. 
# As the threshold increases, the number of matches decreases while match quality improves, reflecting a 
# trade-off between quantity and accuracy. Deterministic matching suffers from high false negatives, 
# while probabilistic matching can introduce false positives and depends on model assumptions and threshold choice.