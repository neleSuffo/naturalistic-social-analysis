library(ggplot2)
library(tidybayes)
library(dplyr)
library(brms)

# Load data
object_data <- read.csv("data/object_test_results.csv")
social_data <- read.csv("data/social_test_results.csv")

# Social Data


social_data_df <- social_data %>%
  mutate(
    child_proximity = if_else(is.na(child_proximity), 0, child_proximity),
    adult_proximity = if_else(is.na(adult_proximity), 0, adult_proximity),
    child_gaze = if_else(is.na(child_gaze), 0, child_gaze),
    adult_gaze = if_else(is.na(adult_gaze), 0, adult_gaze)
  )

# Standardize continuous variables
social_data_df <- social_data_df %>%
  mutate(
    child_proximity = scale(child_proximity),
    adult_proximity = scale(adult_proximity),
    child_gaze = scale(child_gaze),
    adult_gaze = scale(adult_gaze)
  )

# Model with interactions
formula <- person_present ~ child_person + adult_person + 
  child_face + adult_face + 
  child_proximity + adult_proximity + 
  child_gaze + adult_gaze + 
  child_face:child_proximity + adult_face:adult_proximity +
  child_face:child_gaze + adult_face:adult_gaze

# Fit the model (assuming brms for a Bayesian approach)
social_model <- brm(
  formula = formula,
  data = social_data_df,
  family = bernoulli(link = "logit"),
  chains = 8,  # Using 8 chains
  iter = 2000,
  warmup = 1000,
  seed = 123,
  cores = 8,  # Allocating 8 cores for parallelization
  control = list(max_treedepth = 15)
)
# View the model summary (optional)
summary(social_model)

new_data_child <- expand.grid(
  child_face = c(0, 1),                  # Binary: no child face vs. child face present
  child_proximity = seq(-2, 2, length.out = 50),  # Range of standardized child_proximity
  child_person = 0,                      # No child person detected
  adult_person = 0,                      # No adult person detected
  adult_face = 0,                        # No adult face detected
  adult_proximity = 0,                   # Mean of standardized adult_proximity
  child_gaze = 0,                        # Mean of standardized child_gaze
  adult_gaze = 0                         # Mean of standardized adult_gaze
)

# Get predicted probabilities with 95% credible intervals
pred_summary_child <- fitted(
  social_model,
  newdata = new_data_child,
  summary = TRUE,
  probs = c(0.025, 0.975)
)

# Combine predictions with the new data
pred_data_child <- cbind(new_data_child, pred_summary_child)

ggplot(pred_data_child, aes(x = child_proximity, y = Estimate, color = factor(child_face))) +
  geom_line(size = 1) +                          # Line for predicted probability
  geom_ribbon(aes(ymin = Q2.5, ymax = Q97.5, fill = factor(child_face)), alpha = 0.2) +  # Confidence bands
  labs(
    x = "Child Proximity (standardized)",
    y = "Probability of Person Present",
    color = "Child Face",
    fill = "Child Face",
    title = "Effect of Child Proximity and Child Face on Person Presence"
  ) +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = c("blue", "red"), labels = c("No Child Face", "Child Face Present")) +
  scale_fill_manual(values = c("blue", "red"), labels = c("No Child Face", "Child Face Present"))




# Effect of Object Type and Age on Object Presence
# Fit model
object_model_type_age <- brm(
  object_present ~ object_type * age + (1 | ID),
  data = object_data, 
  family = bernoulli(link = "logit"),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  control = list(max_treedepth = 15)
)

# View the model summary (optional)
summary(object_model_type_age)

# Get range of age from your data
age_range <- range(object_data$age, na.rm = TRUE)

# Create prediction grid (marginalized over ID)
new_data_type_age <- expand.grid(
  object_type = levels(object_data$object_type),
  age = seq(age_range[1], age_range[2], length.out = 50)
)

# Get predicted probabilities
pred_summary_type_age <- fitted(object_model_type_age, 
                             newdata = new_data_type_age, 
                             summary = TRUE, 
                             probs = c(0.025, 0.975), 
                             re_formula = NA) # Marginalize over ID

# Combine predictions with new_data
pred_data_type_age <- cbind(new_data_type_age, pred_summary_type_age)

# Clean up object_type names *before* plotting
pred_data_type_age$object_type_clean <- gsub("object_type", "", pred_data_type_age$object_type)

# Plot
ggplot(pred_data_type_age, aes(x = age, y = Estimate, color = object_type_clean)) +
  geom_line(size = 1) +  # Lines for predicted probabilities
  geom_ribbon(aes(ymin = Q2.5, ymax = Q97.5, fill = object_type_clean), 
              alpha = 0.2, color = NA) +  # Shaded 95% CI
  labs(x = "Age", y = "Probability of Object Present",
       color = "Object Type", fill = "Object Type",
       title = "Effect of Object Type and Age on Object Presence") +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1")







# Prepare data
object_data$object_type <- as.factor(object_data$object_type)
object_data$social <- factor(object_data$social, levels = c("alone", "child present", "adult present", "child and adult present"))
object_data$ID <- as.factor(object_data$ID)
# Effect of Object Type and Social on Object Presence
object_model_type_social <- brm(
  object_present ~ object_type * social + (1 | ID),
  data = object_data, 
  family = bernoulli(link = "logit"),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  control = list(max_treedepth = 15)
)

# View the model summary (optional)
summary(object_model_type_social)

# Create prediction grid (marginalized over ID)
new_data_type_social <- expand.grid(
  object_type = levels(object_data$object_type),
  social = levels(object_data$social)  # Include all social levels
)

# Get predicted probabilities
pred_summary_type_social <- fitted(object_model_type_social, 
                                   newdata = new_data_type_social, 
                                   summary = TRUE, 
                                   probs = c(0.025, 0.975), 
                                   re_formula = NA) # Marginalize over ID

# Clean up object_type names before plotting
pred_data_type_social$object_type_clean <- gsub("object_type", "", pred_data_type_social$object_type)

# Plot: Social on x-axis, object_type as color
ggplot(pred_data_type_social, aes(x = social, y = Estimate, color = object_type_clean)) +
  geom_point(position = position_dodge(width = 0.4), size = 3) +
  geom_errorbar(aes(ymin = Q2.5, ymax = Q97.5), 
                width = 0.2, position = position_dodge(width = 0.4)) +
  labs(x = "Social Context", y = "Predicted Probability of Object Presence",
       color = "Object Type",
       title = "Effect of Object Type and Social Context on Object Presence") +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1")





# Fit model with three-way interaction and random intercept for ID
object_model_type_age_social <- brm(
  object_present ~ object_type * age * social + (1 | ID),  # Fixed effects + random intercept
  data = object_data, 
  family = bernoulli(link = "logit"),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  control = list(max_treedepth = 15)
)

# View the model summary (optional)
summary(object_model_type_age_social)

# Get range of age from your data
age_range <- range(object_data$age, na.rm = TRUE)

# Create prediction grid (marginalized over ID)
new_data_type_age_social <- expand.grid(
  object_type = levels(object_data$object_type),
  age = seq(age_range[1], age_range[2], length.out = 50),
  social = levels(object_data$social)
)

# Get predicted probabilities from the correct model
pred_summary_type_age_social <- fitted(
  object_model_type_age_social,
  newdata = new_data_type_age_social,
  summary = TRUE,
  probs = c(0.025, 0.975),
  re_formula = NA  # marginalize over ID
)

# Combine predictions with input grid
pred_data_type_age_social <- cbind(new_data_type_age_social, pred_summary_type_age_social)

# Clean up object_type names
pred_data_type_age_social$object_type_clean <- gsub("object_type", "", pred_data_type_age_social$object_type)

# Plot
ggplot(pred_data_type_age_social, aes(x = age, y = Estimate, color = object_type)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = Q2.5, ymax = Q97.5, fill = object_type), alpha = 0.2, color = NA) +
  facet_wrap(~social, ncol = 2) +  # wrap into 2 columns
  labs(
    x = "Age",
    y = "Predicted Probability of Object Presence",
    color = "Object Type",
    fill = "Object Type",
    title = "Effect of Age and Object Type on Object Presence by Social Context"
  ) +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1")