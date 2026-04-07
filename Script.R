# ==============================================================================
# TITLE:       The Hidden Economic Burden of Disease-Based Inequity:
#              A Synthetic Counterfactual Analysis via Tabular Diffusion Models
# DATA:        Survey of Health, Ageing and Retirement in Europe (SHARE)
# AUTHORS:     Pietro Grassi, Edoardo Paperi, Chiara Seghieri, Daniele Vignoli
# DATE:        April 2026
# ==============================================================================

rm(list=ls())
gc()
set.seed(2026)

# Core Data Manipulation and Imputation Libraries
library(haven)
library(dplyr)
library(tidyr)
library(stringr)
library(purrr)
library(fs)
library(mice) 

# ------------------------------------------------------------------------------
# 0. DIRECTORY SETUP
# ------------------------------------------------------------------------------
input_dir <- "/Users/pietro/Desktop/Sant'Anna/Ricerca/SHARE/DatiMulti"  
output_dir <- "/Users/pietro/Desktop/Sant'Anna/Ricerca/SHARE/EHEW2026" 

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  message(">>> Output directory successfully created: ", output_dir)
}

# ------------------------------------------------------------------------------
# 1. OUTCOME & ECONOMIC EXTRACTION: END-OF-LIFE MODULE (XT)
# ------------------------------------------------------------------------------
files_xt <- dir_ls(path = input_dir, regexp = "sharew.*_xt\\.dta$")
message(">>> Ingesting End-of-Life (XT) modules (Waves 7-9)...")

df_eol <- map_dfr(files_xt, function(f) {
  w <- as.numeric(str_extract(f, "(?<=sharew)\\d+"))
  
  if (!is.na(w) && w %in% c(7, 8, 9)) {
    d <- read_dta(f)
    
    col_palliative <- names(d)[grepl("xt757", names(d))][1]
    col_reason     <- names(d)[grepl("xt754", names(d))][1]
    col_cod        <- names(d)[grepl("xt011", names(d))][1]
    col_oop_costs  <- names(d)[grepl("xt119", names(d))][1] 
    col_help_adl   <- names(d)[grepl("xt022", names(d))][1]
    col_time_help  <- names(d)[grepl("xt024", names(d))][1]
    col_hrs_day    <- names(d)[grepl("xt025", names(d))][1]
    
    d_sel <- d %>%
      select(
        mergeid, country,
        matches("xt020d[1-6]"), matches("xt020d96"), 
        raw_pall_col  = any_of(col_palliative),
        raw_reason    = any_of(col_reason),
        raw_cod       = any_of(col_cod),
        matches("xt119"), 
        raw_help_adl  = any_of(col_help_adl),
        raw_time_help = any_of(col_time_help),
        raw_hrs_day   = any_of(col_hrs_day),
        matches("xt023") 
      ) %>%
      mutate(wave_death = w)
    
    return(d_sel)
  } else { return(tibble()) }
}) %>%
  mutate(across(-c(mergeid), ~as.numeric(haven::zap_labels(.)))) %>%
  mutate(across(-c(mergeid), ~if_else(. < 0, NA_real_, .))) %>%
  mutate(
    # --- Welfare Regime Classification (Extended Esping-Andersen Typology) ---
    welfare_group = case_when(
      country %in% c(13, 18, 55, 14) ~ "Nordic", 
      country %in% c(11, 12, 17, 20, 23, 31) ~ "Continental", 
      country %in% c(15, 16, 19, 33, 53, 59) ~ "Southern", 
      country %in% c(28, 29, 34, 35, 47, 48, 32, 51, 57, 61, 63) ~ "Eastern",
      TRUE ~ NA_character_ 
    ),
    
    # --- Palliative Care Access and Cause of Death Classification ---
    palliative_access = case_when(
      raw_pall_col == 1 ~ 1, 
      raw_pall_col == 5 ~ 0, 
      TRUE ~ NA_real_
    ),
    reason_no_pc = raw_reason,
    cod_trajectory = case_when(
      raw_cod == 1 ~ "Cancer",
      raw_cod %in% c(2, 3, 4, 5) ~ "Organ Failure",
      TRUE ~ "Other"
    ),
    
    # --- Unadjusted Out-of-Pocket (OOP) Costs (Local Currencies) ---
    oop_costs = rowSums(select(., matches("xt119")), na.rm = TRUE),
    oop_costs = if_else(if_all(matches("xt119"), is.na), NA_real_, oop_costs),
    
    # --- Informal Caregiving Burden Assessment ---
    help_received = case_when(raw_help_adl == 1 ~ 1, raw_help_adl == 5 ~ 0, TRUE ~ NA_real_),
    only_formal_help = if_else(
      rowSums(select(., matches("xt023")), na.rm = TRUE) > 0 & 
        coalesce(xt023d14, 0) == 1 &                             
        rowSums(select(., matches("xt023d[1-9]|xt023d1[0-3]|xt023d1[5-6]")), na.rm = TRUE) == 0, 
      1, 0
    ),
    time_help_cat = case_when(help_received == 0 | only_formal_help == 1 ~ 0, help_received == 1 ~ as.numeric(raw_time_help), TRUE ~ NA_real_),
    help_hrs_day_cont = case_when(help_received == 0 | only_formal_help == 1 ~ 0, help_received == 1 ~ as.numeric(raw_hrs_day), TRUE ~ NA_real_),
    eol_adl_score = rowSums(select(., matches("xt020d[1-6]"), matches("xt020d96")), na.rm = TRUE),
    eol_adl_score = if_else(if_all(c(matches("xt020d[1-6]"), matches("xt020d96")), is.na), NA_real_, eol_adl_score)
  ) %>%
  filter(palliative_access == 1 | (palliative_access == 0 & reason_no_pc %in% c(2, 3))) %>%
  filter(!is.na(palliative_access)) %>%
  filter(!is.na(welfare_group)) %>%
  mutate(welfare_group = factor(welfare_group, levels = c("Nordic", "Continental", "Southern", "Eastern")))

message(">>> Total target deceased population identified: ", nrow(df_eol))

# ------------------------------------------------------------------------------
# 2. SOCIO-ECONOMIC & DEMOGRAPHIC EXTRACTION (GV_IMPUTATIONS)
# ------------------------------------------------------------------------------
message(">>> Ingesting SHARE gv_imputations datasets...")
files_gv_imp <- dir_ls(path = input_dir, regexp = "sharew.*_gv_imputations\\.dta$")

df_imputations <- map_dfr(files_gv_imp, function(f) {
  w <- as.numeric(str_extract(f, "(?<=sharew)\\d+"))
  if (!is.na(w) && w %in% c(6, 7, 8)) {
    read_dta(f) %>%
      select(mergeid, implicat, thinc2, hnetw, isced, age, gender, mstat, nchild, exrate, 
             otrf, fdistress, chronic) %>%
      mutate(wave = w)
  } else { return(tibble()) }
}) %>%
  filter(implicat == 1)

# ------------------------------------------------------------------------------
# 3. DATA LINKAGE AND CURRENCY HARMONIZATION
# ------------------------------------------------------------------------------
message(">>> Merging modules and harmonizing currencies to Euro...")
last_wave_imp <- df_imputations %>%
  inner_join(select(df_eol, mergeid, wave_death), by = "mergeid") %>%
  filter(wave < wave_death) %>% 
  group_by(mergeid) %>%
  summarise(max_wave = max(wave))

df_imputations_last <- df_imputations %>%
  inner_join(last_wave_imp, by = c("mergeid", "wave" = "max_wave")) %>%
  mutate(
    nchild = if_else(nchild < 0, NA_real_, nchild)
  )

df_raw_merged <- df_eol %>%
  inner_join(df_imputations_last, by = "mergeid") %>%
  mutate(
    exrate = coalesce(exrate, 1),
    oop_costs = oop_costs / exrate,
    owner = if_else(otrf == 1, 1, 0),
    fdistress_inv = 5 - fdistress 
  ) %>%
  select(
    mergeid, country, wave_death,
    age, gender, isced, mstat, nchild, thinc2, hnetw, eol_adl_score, 
    cod_trajectory, welfare_group, palliative_access, 
    owner, fdistress_inv, chronic,
    time_help_cat, help_hrs_day_cont, oop_costs                                  
  )

# ------------------------------------------------------------------------------
# 4. MISSINGNESS AUDIT 
# ------------------------------------------------------------------------------
message(">>> Performing Missingness Audit on raw merged cohort...")

missing_pct <- sapply(df_raw_merged, function(x) sum(is.na(x)) / nrow(df_raw_merged) * 100)
missing_df <- data.frame(Variable = names(missing_pct), Missing_Pct = round(missing_pct, 2))
print(missing_df)

df_filtered <- df_raw_merged

# ------------------------------------------------------------------------------
# 5. FULL MICE IMPUTATION & FMI AUDIT (Madley-Dowd, 2019 Framework)
# ------------------------------------------------------------------------------
message(">>> Performing Multivariate Imputation by Chained Equations (MICE)...")

# Prepare dataframe for imputation
df_for_mice <- df_filtered %>% 
  select(-mergeid) %>%
  mutate(across(where(haven::is.labelled), ~ as.numeric(haven::zap_labels(.)))) %>%
  mutate(across(where(is.character), as.factor))

# Multiple imputation (m=10 required for accurate FMI calculation)
m_imputations <- 10
imputed_data <- mice(df_for_mice, m = m_imputations, method = 'pmm', seed = 2026, printFlag = FALSE)

message(">>> Extracting Fraction of Missing Information (FMI) for core outcomes...")

# FMI is parameter-specific. Fit baseline diagnostic models across the m imputations
fit_oop <- with(imputed_data, lm(oop_costs ~ age + gender + eol_adl_score))
fit_hrs <- with(imputed_data, lm(help_hrs_day_cont ~ age + gender + eol_adl_score))

pool_oop <- pool(fit_oop)
pool_hrs <- pool(fit_hrs)

# Extract FMI for model coefficients
fmi_oop <- summary(pool_oop, type = "all") %>% select(term, fmi) %>% mutate(Model = "OOP Costs Diagnostic")
fmi_hrs <- summary(pool_hrs, type = "all") %>% select(term, fmi) %>% mutate(Model = "Care Hours Diagnostic")

fmi_table <- bind_rows(fmi_oop, fmi_hrs) %>%
  select(Model, term, fmi) %>%
  mutate(fmi_pct = round(fmi * 100, 2))

print(fmi_table)

# Calculate average FMI across models
mean_fmi <- mean(fmi_table$fmi, na.rm = TRUE)
message(sprintf(">>> AVERAGE FMI ACROSS DIAGNOSTIC MODELS: %.2f%%", mean_fmi * 100))

# Extract the first imputed dataset to train the TabDDPM generative model
df_complete <- complete(imputed_data, 1)

df_final <- df_complete %>%
  mutate(
    gender = if_else(gender == 2, "Female", "Male"),
    mstat = if_else(mstat == 1, "Partnered", "Single"),
    covid_period = factor(if_else(wave_death == 9, "During COVID-19", "Pre-COVID-19"), 
                          levels = c("Pre-COVID-19", "During COVID-19"))
  )

message(">>> Imputation complete. Total observations ready for Python TabDDPM: ", nrow(df_final))

# ------------------------------------------------------------------------------
# 6. EXPORT TO PYTHON
# ------------------------------------------------------------------------------
output_file <- file.path(output_dir, "share_prepared_for_tabddpm.csv")
write.csv(df_final %>% select(-wave_death, -covid_period), output_file, row.names = FALSE)
message(">>> Export complete. ", nrow(df_final), " observations ready for TabDDPM synthetic generation.")

# ------------------------------------------------------------------------------
# 7. LOAD COUNTERFACTUAL DATA, PPP ADJUSTMENT & MONETIZATION
# ------------------------------------------------------------------------------
message(">>> Importing Synthetic Counterfactuals generated by TabDDPM...")

# Loading Econometric and Data Visualization Libraries
if(!require(estimatr)) install.packages("estimatr")
if(!require(quantreg)) install.packages("quantreg")
if(!require(modelsummary)) install.packages("modelsummary")
if(!require(boot)) install.packages("boot")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(broom)) install.packages("broom")

library(estimatr); library(quantreg); library(modelsummary)
library(boot); library(ggplot2); library(broom); library(dplyr)

cf_file <- file.path(output_dir, "synthetic_counterfactuals_Q1.csv")
df_cf <- read.csv(cf_file)
df_cf$covid_period <- df_final$covid_period

# ------------------------------------------------------------------------------
# 7.1 DYNAMIC PURCHASING POWER PARITY (PPP) ADJUSTMENT
# ------------------------------------------------------------------------------
message(">>> Computing dynamic PPP adjustment from Eurostat dataset...")

# Source: Eurostat (prc_ppp_ind). Indicator: PLI for Actual Individual Consumption.
# The index is calculated as a 5-year moving average across Waves 7-9 (2017-2021).
eurostat_file <- file.path(input_dir, "prc_ppp_ind$defaultview_linear_2_0.csv")

# SHARE country code to Eurostat ISO mapping
share_iso_map <- data.frame(
  country = c(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 28, 29, 31, 32, 33, 34, 35, 47, 48, 51, 53, 55, 57, 59, 61, 63),
  geo = c("AT", "DE", "SE", "NL", "ES", "IT", "FR", "DK", "EL", "CH", "BE", "CZ", "PL", "LU", "HU", "PT", "SI", "EE", "HR", "LT", "BG", "CY", "FI", "LV", "MT", "RO", "SK")
)

# Computing baseline PPP indices (EU27 = 1.0)
df_eurostat <- read.csv(eurostat_file, stringsAsFactors = FALSE)

ppp_indices <- df_eurostat %>%
  filter(
    na_item == "PLI_EU27_2020",    
    ppp_cat == "A01",              
    TIME_PERIOD %in% 2017:2021     
  ) %>%
  group_by(geo) %>%
  summarise(
    ppp_idx = mean(OBS_VALUE, na.rm = TRUE) / 100, 
    .groups = 'drop'
  ) %>%
  inner_join(share_iso_map, by = "geo") %>%
  select(country, ppp_idx)

# Merging PPP adjustment to the counterfactual dataset
df_cf <- df_cf %>%
  left_join(ppp_indices, by = "country") %>%
  mutate(ppp_idx = coalesce(ppp_idx, 1)) 

# ------------------------------------------------------------------------------
# 7.2 ECONOMIC MONETIZATION & VARIANCE REDUCTION
# ------------------------------------------------------------------------------
message(">>> Applying country-specific shadow wages (Proxy Good Method)...")

# Dynamic ingestion of Eurostat labor cost microdata
lci_file <- file.path(input_dir, "lc_lci_lev$defaultview_linear_2_0.csv")
df_lci <- read.csv(lci_file, stringsAsFactors = FALSE)

# Automated extraction of shadow wages (NACE Rev.2 Sector proxy)
shadow_wages <- df_lci %>%
  filter(
    TIME_PERIOD == 2020,  
    nace_r2 == "B-S_X_O", 
    unit == "EUR"         
  ) %>%
  filter(lcstruct %in% c("TOTAL", "D1", "D11")) %>%
  arrange(geo, lcstruct) %>%
  group_by(geo) %>%
  slice(1) %>%
  ungroup() %>%
  select(geo, hourly_wage = OBS_VALUE) %>%
  inner_join(share_iso_map, by = "geo") %>%
  select(country, hourly_wage)

df_cf <- df_cf %>%
  left_join(shadow_wages, by = "country") %>%
  mutate(hourly_wage = coalesce(as.numeric(hourly_wage), 15.0)) # Fallback imputation for missing sector wages

message(">>> Applying physiological capping and 1% winsorization to bound variance...")

# Physiological Capping (Max 16 waking hours/day equivalent)
MAX_HOURS_YEAR <- 16 * 365

# 1st and 99th Percentile Winsorization function to mitigate extreme fat-tail outliers
winsorize_1pct <- function(x) {
  q_low <- quantile(x, 0.01, na.rm = TRUE)
  q_high <- quantile(x, 0.99, na.rm = TRUE)
  pmin(pmax(x, q_low), q_high)
}

df_econ <- df_cf %>%
  mutate(
    # Cap informal care at physiological limits
    real_care_hours = pmin(real_care_hours, MAX_HOURS_YEAR),
    synth_care_hours = pmin(synth_care_hours, MAX_HOURS_YEAR),
    
    # OOP Costs standardizing
    real_oop_costs_ppp  = real_oop_costs / ppp_idx,
    synth_oop_costs_ppp = synth_oop_costs / ppp_idx,
    
    # Nominal Shadow Pricing
    real_informal_nom  = real_care_hours * hourly_wage,
    synth_informal_nom = synth_care_hours * hourly_wage,
    
    # Shadow Price standardizing (PPS Euro)
    real_informal_ppp  = real_informal_nom / ppp_idx,
    synth_informal_ppp = synth_informal_nom / ppp_idx,
    
    # Net Economic Burden (PPS Euro)
    real_total_burden  = real_oop_costs_ppp + real_informal_ppp,
    synth_total_burden = synth_oop_costs_ppp + synth_informal_ppp,
    
    # Individual Treatment Effects (ITE)
    delta_oop          = synth_oop_costs_ppp - real_oop_costs_ppp,
    delta_hours        = synth_care_hours - real_care_hours,
    delta_total_burden = synth_total_burden - real_total_burden
  ) %>%
  mutate(
    # Variance reduction via robust winsorization on treatment effects
    delta_oop          = winsorize_1pct(delta_oop),
    delta_hours        = winsorize_1pct(delta_hours),
    delta_total_burden = winsorize_1pct(delta_total_burden),
    
    # Feature stratification
    wealth_quartile = factor(ntile(hnetw, 4), labels = c("Q1 (Poorest)", "Q2", "Q3", "Q4 (Richest)")),
    cod_trajectory  = factor(cod_trajectory, levels = c("Cancer", "Organ Failure", "Other"))
  )

# ------------------------------------------------------------------------------
# 7.3 DESCRIPTIVE STATISTICS (TABLE 1: EMPIRICAL VS ATE SCENARIOS)
# ------------------------------------------------------------------------------
message(">>> Generating Table 1 (Descriptive Statistics for Empirical and ATE Cohorts)...")

# ==============================================================================
# 1. EMPIRICAL COHORT (Observed Reality)
# ==============================================================================
df_desc_emp <- df_econ %>%
  mutate(
    Palliative_Care = if_else(palliative_access == 1, "Treated", "Untreated"),
    Age = as.numeric(age),
    Gender = as.factor(gender),
    Marital_Status = as.factor(mstat),
    Education_ISCED = as.factor(isced),
    Children_N = as.numeric(nchild),
    Wealth_Quartile = as.factor(wealth_quartile),
    Welfare_Regime = as.factor(welfare_group),
    Cause_of_Death = as.factor(cod_trajectory),
    ADL_Dependency_Score = as.numeric(eol_adl_score),
    Homeowner = factor(if_else(owner == 1, "Yes", "No"), levels = c("No", "Yes")),
    Financial_Distress = as.numeric(fdistress_inv),
    Comorbidities_N = as.numeric(chronic),
    
    Shadow_Hourly_Wage_Euro = as.numeric(hourly_wage),
    Observed_OOP_Costs_PPS = as.numeric(if_else(palliative_access == 1, synth_oop_costs_ppp, real_oop_costs_ppp)),
    Observed_Care_Hours = as.numeric(if_else(palliative_access == 1, synth_care_hours, real_care_hours))
  ) %>%
  select(Palliative_Care, Age, Gender, Marital_Status, Education_ISCED, Children_N, 
         Homeowner, Financial_Distress, Comorbidities_N, 
         Wealth_Quartile, Welfare_Regime, Cause_of_Death, ADL_Dependency_Score, 
         Shadow_Hourly_Wage_Euro, Observed_OOP_Costs_PPS, Observed_Care_Hours)

# ==============================================================================
# 2. ATE COUNTERFACTUAL COHORT (Universal Scenarios)
# ==============================================================================
df_desc_ate_y0 <- df_econ %>% 
  mutate(
    Scenario = "Standard Care (Y0)", 
    Potential_OOP_Costs_PPS = as.numeric(real_oop_costs_ppp), 
    Potential_Care_Hours = as.numeric(real_care_hours)
  )

df_desc_ate_y1 <- df_econ %>% 
  mutate(
    Scenario = "Palliative Care (Y1)", 
    Potential_OOP_Costs_PPS = as.numeric(synth_oop_costs_ppp), 
    Potential_Care_Hours = as.numeric(synth_care_hours)
  )

df_desc_ate <- bind_rows(df_desc_ate_y0, df_desc_ate_y1) %>%
  mutate(
    Scenario = factor(Scenario, levels = c("Standard Care (Y0)", "Palliative Care (Y1)")),
    Age = as.numeric(age),
    Gender = as.factor(gender),
    Marital_Status = as.factor(mstat),
    Education_ISCED = as.factor(isced),
    Children_N = as.numeric(nchild),
    Wealth_Quartile = as.factor(wealth_quartile),
    Welfare_Regime = as.factor(welfare_group),
    Cause_of_Death = as.factor(cod_trajectory),
    ADL_Dependency_Score = as.numeric(eol_adl_score),
    
    Homeowner = factor(if_else(owner == 1, "Yes", "No"), levels = c("No", "Yes")),
    Financial_Distress = as.numeric(fdistress_inv),
    Comorbidities_N = as.numeric(chronic),
    
    Shadow_Hourly_Wage_Euro = as.numeric(hourly_wage)
  ) %>%
  select(Scenario, Age, Gender, Marital_Status, Education_ISCED, Children_N, 
         Homeowner, Financial_Distress, Comorbidities_N, 
         Wealth_Quartile, Welfare_Regime, Cause_of_Death, ADL_Dependency_Score, 
         Shadow_Hourly_Wage_Euro, Potential_OOP_Costs_PPS, Potential_Care_Hours)

# ==============================================================================
# 3. EXPORT VIA DATASUMMARY_BALANCE
# ==============================================================================
datasummary_balance(
  ~ Palliative_Care,
  data = df_desc_emp,
  dinm = FALSE,  
  title = "Table 1a: Baseline Characteristics and Observed Outcomes (Empirical Cohort)",
  output = file.path(output_dir, "Table_1a_Empirical_Descriptives.docx")
)

datasummary_balance(
  ~ Scenario,
  data = df_desc_ate,
  dinm = FALSE,
  title = "Table 1b: Potential Outcomes and Demographics (Full ATE Digital Twin Cohort)",
  output = file.path(output_dir, "Table_1b_ATE_Descriptives.docx")
)

message(">>> Tables 1a and 1b successfully exported.")

# ------------------------------------------------------------------------------
# 8. NON-PARAMETRIC ATE (BIAS-CORRECTED BOOTSTRAPPED)
# ------------------------------------------------------------------------------
message(">>> Computing BCa Bootstrapped ATE (1000 iterations)...")
mean_boot <- function(data, indices) { return(mean(data[indices], na.rm=TRUE)) }

boot_oop   <- boot(df_econ$delta_oop, mean_boot, R = 1000)
boot_hours <- boot(df_econ$delta_hours, mean_boot, R = 1000)
boot_total <- boot(df_econ$delta_total_burden, mean_boot, R = 1000)

# Applying Bias-Corrected and accelerated (BCa) bootstrap to account for severe healthcare cost skewness.
ci_oop   <- boot.ci(boot_oop, type="bca")
ci_hours <- boot.ci(boot_hours, type="bca")
ci_total <- boot.ci(boot_total, type="bca")

cat("\n=================================================================\n")
cat("   BOOTSTRAPPED CAUSAL ATE RESULTS (PPP ADJUSTED, BCa 95% CI)\n")
cat("=================================================================\n")
cat(sprintf("ATE OOP Costs (PPS Euro):  € %8.2f [%.2f, %.2f]\n", boot_oop$t0, ci_oop$bca[4], ci_oop$bca[5]))
cat(sprintf("ATE Informal Care Hours:     %8.1f hrs [%.1f, %.1f]\n", boot_hours$t0, ci_hours$bca[4], ci_hours$bca[5]))
cat(sprintf("ATE Net Total Burden:      € %8.2f [%.2f, %.2f]\n", boot_total$t0, ci_total$bca[4], ci_total$bca[5]))
cat("=================================================================\n")

# ------------------------------------------------------------------------------
# 9. MULTIVARIATE QUANTILE TREATMENT EFFECTS (QTE)
# ------------------------------------------------------------------------------
message(">>> Estimating Distributional Causal Effects via Quantile Regression...")

# Applying micro-jittering to resolve singular design matrices caused by zero-inflation ties in categorical covariates.
df_econ <- df_econ %>%
  mutate(delta_total_burden_jitter = delta_total_burden + runif(n(), min = -1e-4, max = 1e-4))

formula_qte <- delta_total_burden_jitter ~ cod_trajectory + welfare_group + wealth_quartile + age + gender + mstat + eol_adl_score + owner + fdistress_inv + chronic + covid_period

# Estimating conditional quantile models
qte_50 <- suppressWarnings(rq(formula_qte, tau = 0.50, data = df_econ)) 
qte_75 <- suppressWarnings(rq(formula_qte, tau = 0.75, data = df_econ)) 
qte_90 <- suppressWarnings(rq(formula_qte, tau = 0.90, data = df_econ)) 

qte_models <- list("Median (50th)" = qte_50, "Severe (75th)" = qte_75, "Catastrophic (90th)" = qte_90)

message(">>> Compiling Table 3 (Bootstrapped standard errors in progress)...")

# Exporting Table 3. Warnings suppressed to allow asymptotic fallback computation via the broom package.
suppressWarnings({
  modelsummary(
    qte_models,
    estimate  = "{estimate}{stars}",
    statistic = "conf.int",
    se.type   = "nid",
    gof_omit  = "IC|Log|Adj|p\\.value|F|RMSE",
    title     = "Table 3: Multivariate Quantile Treatment Effects (QTE) on Net Economic Burden",
    output    = file.path(output_dir, "Table_3_QTE_Models.docx")
  )
})
message(">>> Table 3 (QTE Models) successfully exported.")

# ------------------------------------------------------------------------------
# 10. MULTIVARIATE CATE ESTIMATION (CLUSTERED ROBUST OLS)
# ------------------------------------------------------------------------------
message(">>> Modeling Conditional Average Treatment Effects...")

formula_cate_oop   <- delta_oop ~ cod_trajectory + welfare_group + wealth_quartile + age + gender + mstat + eol_adl_score + owner + fdistress_inv + chronic + covid_period
formula_cate_hours <- delta_hours ~ cod_trajectory + welfare_group + wealth_quartile + age + gender + mstat + eol_adl_score + owner + fdistress_inv + chronic + covid_period
formula_cate_total <- delta_total_burden ~ cod_trajectory + welfare_group + wealth_quartile + age + gender + mstat + eol_adl_score + owner + fdistress_inv + chronic + covid_period

# Implementing CR2 Cluster-Robust Standard Errors to account for small intra-national correlation (N_clusters < 40).
model_oop   <- lm_robust(formula_cate_oop, data = df_econ, clusters = country, se_type = "CR2")
model_hours <- lm_robust(formula_cate_hours, data = df_econ, clusters = country, se_type = "CR2")
model_total <- lm_robust(formula_cate_total, data = df_econ, clusters = country, se_type = "CR2")

models_list <- list("Δ OOP Costs (PPS €)" = model_oop, "Δ Care (Hours)" = model_hours, "Δ Net Burden (PPS €)" = model_total)

modelsummary(
  models_list,
  estimate  = "{estimate}{stars}",
  statistic = "conf.int",
  gof_omit  = "IC|Log|Adj|p\\.value|F|RMSE",
  title     = "Table 2: Multivariate Analysis of CATE (CR2 Cluster-Robust Standard Errors)",
  output    = file.path(output_dir, "Table_2_CATE_Models.docx")
)

# ------------------------------------------------------------------------------
# 11.1 PUBLICATION-READY VISUALIZATION: CATE FOREST PLOT
# ------------------------------------------------------------------------------
message(">>> Generating Marginal CATE Forest Plot...")

coef_df <- tidy(model_total) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    term = str_replace_all(term, "cod_trajectory", "Disease: "),
    term = str_replace_all(term, "wealth_quartile", "Wealth: "),
    term = str_replace_all(term, "welfare_group", "Welfare: "),
    term = str_replace_all(term, "genderMale", "Gender: Male"),
    term = str_replace_all(term, "age", "Age"),
    term = str_replace_all(term, "mstatSingle", "Marital: Single"),
    term = str_replace_all(term, "eol_adl_score", "ADL Dependency Score"),
    term = str_replace_all(term, "fdistress_inv", "Financial Distress"),
    term = str_replace_all(term, "owner", "Home Owner: Yes"),
    term = str_replace_all(term, "chronic", "Comorbidities"),
    term = str_replace_all(term, "covid_periodDuring COVID-19", "COVID-19 Pandemic"),
    term = factor(term, levels = rev(unique(term)))
  )

p_forest <- ggplot(coef_df, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#c0392b", linewidth = 1) +
  geom_point(color = "#2c3e50", size = 3.5) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "#2c3e50", linewidth = 0.8) +
  labs(
    x = "Effect Size in PPS Euros (€)", y = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(panel.grid.minor = element_blank(),
        axis.text.y = element_text(face = "bold", color = "black", size=11))

ggsave(file.path(output_dir, "Figure_1_CATE_ForestPlot.png"), plot = p_forest, width = 10, height = 6, dpi = 300, bg="white")

# ------------------------------------------------------------------------------
# 11.2 PUBLICATION-READY VISUALIZATION: CAUSAL SHIFT DISTRIBUTIONS (KDE)
# ------------------------------------------------------------------------------
message(">>> Generating Distributional Causal Shift Density Plots...")

if(!require(scales)) install.packages("scales")
library(scales)

# Wrapper function for rendering KDE overlapping distributions
plot_causal_shift <- function(data, col_y0, col_y1, x_label, file_name) {
  
  # Reshaping to long format
  df_long <- data %>%
    select(all_of(c(col_y0, col_y1))) %>%
    pivot_longer(cols = everything(), names_to = "Cohort", values_to = "Value") %>%
    mutate(
      Cohort = if_else(Cohort == col_y0, "Standard Care (Y0)", "Palliative Care (Y1)"),
      Cohort = factor(Cohort, levels = c("Standard Care (Y0)", "Palliative Care (Y1)"))
    )
  
  # Truncating extreme outliers
  p95 <- quantile(df_long$Value, 0.99, na.rm = TRUE)
  df_plot <- df_long %>% filter(Value <= p95)
  
  # ----------------------------------------------------------------------------
  # Dynamic Y-axis scaling (Scientific notation & custom breaks)
  # ----------------------------------------------------------------------------
  val_y0 <- df_plot$Value[df_plot$Cohort == "Standard Care (Y0)"]
  val_y1 <- df_plot$Value[df_plot$Cohort == "Palliative Care (Y1)"]
  
  dens_y0 <- density(val_y0, adjust = 2, na.rm = TRUE)
  dens_y1 <- density(val_y1, adjust = 2, na.rm = TRUE)
  max_dens <- max(c(dens_y0$y, dens_y1$y))
  
  exponent <- floor(log10(max_dens))
  power_val <- 10^exponent
  
  # Compute manual breaks
  max_scaled_val <- ceiling(max_dens / power_val)
  y_breaks <- seq(0, max_scaled_val, by = 1) 
  y_breaks_real <- y_breaks * power_val     
  # ----------------------------------------------------------------------------
  
  p_density <- ggplot(df_plot, aes(x = Value, fill = Cohort, color = Cohort)) +
    geom_density(alpha = 0.4, linewidth = 1.2, adjust = 2) +
    
    scale_fill_manual(values = c("Standard Care (Y0)" = "#2c3e50", "Palliative Care (Y1)" = "#e74c3c")) +
    scale_color_manual(values = c("Standard Care (Y0)" = "#2c3e50", "Palliative Care (Y1)" = "#e74c3c")) +
    
    scale_x_continuous(
      labels = comma, 
      breaks = function(x) { b <- pretty(x, n = 6); b[b >= 0] }, 
      expand = c(0, 0)
    ) +
    
    # Y-axis: Enforce limits beyond breaks to ensure the last label renders
    scale_y_continuous(
      breaks = y_breaks_real,
      limits = c(0, max(y_breaks_real)), 
      labels = function(x) x / power_val,
      expand = expansion(mult = c(0, 0.10)) # 10% upper expansion
    ) +
    
    coord_cartesian(xlim = c(-p95 * 0.05, p95)) +
    
    # Render mathematical exponent
    labs(
      x = x_label, 
      y = parse(text = paste0("Density~Estimation~(10^", exponent, ")"))
    ) +
    
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "top", 
      legend.title = element_blank(),
      
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(), 
      
      axis.line.x = element_line(color = "black", linewidth = 0.6),
      axis.ticks.x = element_line(color = "black"),
      
      axis.line.y = element_line(color = "black", linewidth = 0.6),
      axis.ticks.y = element_line(color = "black"),
      axis.text.y = element_text(color = "black"),
      
      plot.margin = margin(t = 15, r = 20, b = 15, l = 15)
    )
  
  ggsave(file.path(output_dir, file_name), plot = p_density, width = 9, height = 6, dpi = 300, bg="white")
}

plot_causal_shift(
  df_econ, 
  col_y0 = "real_oop_costs_ppp", col_y1 = "synth_oop_costs_ppp", 
  x_label = "OOP Costs (PPS €)", 
  file_name = "Figure_2_Shift_OOP.png"
)

plot_causal_shift(
  df_econ, 
  col_y0 = "real_care_hours", col_y1 = "synth_care_hours", 
  x_label = "Total Annual Caregiving Hours", 
  file_name = "Figure_3_Shift_Hours.png"
)

plot_causal_shift(
  df_econ, 
  col_y0 = "real_informal_ppp", col_y1 = "synth_informal_ppp", 
  x_label = "Shadow Value of Informal Care (PPS €)", 
  file_name = "Figure_4_Shift_Informal_Value.png"
)

message(">>> Distributional causal shift plots successfully exported.")

# ------------------------------------------------------------------------------
# 11.3 PUBLICATION-READY VISUALIZATION: CATE & QTE INEQUITY DYNAMICS
# ------------------------------------------------------------------------------
message(">>> Generating Inequity Visualizations (CATE & QTE)...")

# ==============================================================================
# PLOT A: INSTITUTIONAL INEQUITY (CATE Eastern Penalty across outcomes)
# ==============================================================================
# Extract Eastern Europe coefficient from CATE models
tidy_cate_oop <- tidy(model_oop, conf.int = TRUE) %>% filter(term == "welfare_groupEastern") %>% mutate(Outcome = "Δ OOP Costs (PPS €)")
tidy_cate_hrs <- tidy(model_hours, conf.int = TRUE) %>% filter(term == "welfare_groupEastern") %>% mutate(Outcome = "Δ Care (Hours)")
tidy_cate_tot <- tidy(model_total, conf.int = TRUE) %>% filter(term == "welfare_groupEastern") %>% mutate(Outcome = "Δ Net Burden (PPS €)")

cate_eastern_df <- bind_rows(tidy_cate_oop, tidy_cate_hrs, tidy_cate_tot) %>%
  mutate(Outcome = factor(Outcome, levels = c("Δ OOP Costs (PPS €)", "Δ Care (Hours)", "Δ Net Burden (PPS €)")))

p_cate_eastern <- ggplot(cate_eastern_df, aes(x = Outcome, y = estimate, fill = Outcome)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_col(width = 0.6, color = "black", alpha = 0.85) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.15, linewidth = 0.8) +
  scale_fill_manual(values = c("Δ OOP Costs (PPS €)" = "#c0392b", "Δ Care (Hours)" = "#f39c12", "Δ Net Burden (PPS €)" = "#8e44ad")) +
  facet_wrap(~ Outcome, scales = "free_y") + # Facet plots to accommodate different scales (Currency vs. Hours)
  scale_y_continuous(labels = comma) +
  labs(
    x = "",
    y = "Average Penalty (PPS € / Hours)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 13),
    axis.text.x = element_blank(), # Remove redundant X-axis labels
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.line = element_line(color = "black", linewidth = 0.6),
    axis.ticks.y = element_line(color = "black")
  )

ggsave(file.path(output_dir, "Figure_5_CATE_Eastern.png"), plot = p_cate_eastern, width = 10, height = 5, dpi = 300, bg="white")

# ==============================================================================
# PLOT B: CLINICAL INEQUITY (QTE Disease Dynamics)
# ==============================================================================
# Extract coefficients from QTE models
suppressWarnings({
  tidy_qte_50 <- tidy(qte_50, se.type = "nid", conf.int = TRUE) %>% mutate(Quantile = "50th (Median)", tau = 0.50)
  tidy_qte_75 <- tidy(qte_75, se.type = "nid", conf.int = TRUE) %>% mutate(Quantile = "75th (Severe)", tau = 0.75)
  tidy_qte_90 <- tidy(qte_90, se.type = "nid", conf.int = TRUE) %>% mutate(Quantile = "90th (Catastrophic)", tau = 0.90)
})

qte_clinical_df <- bind_rows(tidy_qte_50, tidy_qte_75, tidy_qte_90) %>%
  filter(term %in% c("cod_trajectoryOrgan Failure", "cod_trajectoryOther")) %>%
  mutate(
    term = case_when(
      term == "cod_trajectoryOrgan Failure" ~ "Organ Failure",
      term == "cod_trajectoryOther" ~ "Other (Frailty/Dementia)"
    ),
    Quantile = factor(Quantile, levels = c("50th (Median)", "75th (Severe)", "90th (Catastrophic)"))
  )

p_qte_clinical <- ggplot(qte_clinical_df, aes(x = Quantile, y = estimate, group = term, color = term)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 4.5) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.15, linewidth = 0.8) +
  scale_color_manual(values = c("Organ Failure" = "#e67e22", "Other (Frailty/Dementia)" = "#8e44ad")) +
  scale_y_continuous(labels = comma) +
  labs(
    x = "Severity Percentile (Total Economic Burden)",
    y = "Additional Economic Penalty (PPS €)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", linewidth = 0.6),
    axis.ticks = element_line(color = "black")
  )

ggsave(file.path(output_dir, "Figure_6_QTE_Clinical.png"), plot = p_qte_clinical, width = 8, height = 6, dpi = 300, bg="white")

message(">>> CATE & QTE Inequity Visualizations successfully exported.")

# ==============================================================================
# 12. METHODOLOGICAL APPENDIX: SENSITIVITY ANALYSES
# ==============================================================================
# Evaluating net burden sensitivity to local market wage scalars (50% to 150%)
message(">>> Evaluating net burden sensitivity (Bootstrapped CIs)...")
wage_multipliers <- c(0.50, 0.75, 1.00, 1.25, 1.50)
sensitivity_results <- data.frame(Multiplier = numeric(), Net_Burden = numeric(), CI_Lower = numeric(), CI_Upper = numeric())

mean_boot <- function(data, indices) { return(mean(data[indices], na.rm=TRUE)) }

for (m in wage_multipliers) {
  df_sens <- df_econ %>%
    mutate(
      sens_delta_total = delta_oop + ((delta_hours * (hourly_wage * m)) / ppp_idx)
    )
  
  # Consistent bootstrapping procedure
  boot_sens <- boot(df_sens$sens_delta_total, mean_boot, R = 1000)
  ci_sens <- boot.ci(boot_sens, type = "bca")
  
  sensitivity_results <- rbind(sensitivity_results, data.frame(
    Multiplier = m * 100, 
    Net_Burden = boot_sens$t0, 
    CI_Lower = ci_sens$bca[4], 
    CI_Upper = ci_sens$bca[5]
  ))
}

p_sens <- ggplot(sensitivity_results, aes(x = Multiplier, y = Net_Burden)) + 
  geom_ribbon(aes(ymin=CI_Lower, ymax=CI_Upper), fill="#2980b9", alpha=0.2) +
  geom_line(color="#2980b9", linewidth=1.2) +
  geom_point(color="#2c3e50", size=4) +
  geom_hline(yintercept=0, linetype="dashed", color="#c0392b") +
  scale_x_continuous(breaks = c(50, 75, 100, 125, 150)) + 
  labs(
    x = "Market Wage Multiplier (%)", y = "Net ATE (PPS €)"
  ) +
  theme_minimal(base_size = 14)

ggsave(file.path(output_dir, "Appendix_Fig1_WageSensitivity.png"), plot = p_sens, width = 8, height = 5, dpi = 300, bg="white")

# ==============================================================================
# 12.2 METHODOLOGICAL APPENDIX: COVID-19 ROBUSTNESS CHECK
# ==============================================================================
message(">>> Running COVID-19 Time-Period Robustness Check...")

formula_cate_split <- delta_total_burden ~ cod_trajectory + welfare_group + wealth_quartile + age + gender + mstat + eol_adl_score + owner + fdistress_inv + chronic

model_total_pre <- lm_robust(
  formula_cate_split, 
  data = subset(df_econ, covid_period == "Pre-COVID-19"), 
  clusters = country, se_type = "CR2"
)

model_total_covid <- lm_robust(
  formula_cate_split, 
  data = subset(df_econ, covid_period == "During COVID-19"), 
  clusters = country, se_type = "CR2"
)

# Extract and format coefficients for visualization
tidy_pre <- tidy(model_total_pre) %>% mutate(Period = "Pre-COVID-19 (2016-2019)")
tidy_covid <- tidy(model_total_covid) %>% mutate(Period = "During COVID-19 (2020-2021)")

coef_covid_df <- bind_rows(tidy_pre, tidy_covid) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    term = str_replace_all(term, "cod_trajectory", "Disease: "),
    term = str_replace_all(term, "wealth_quartile", "Wealth: "),
    term = str_replace_all(term, "welfare_group", "Welfare: "),
    term = str_replace_all(term, "genderMale", "Gender: Male"),
    term = str_replace_all(term, "mstatSingle", "Marital: Single"),
    term = str_replace_all(term, "age", "Age"),
    term = str_replace_all(term, "eol_adl_score", "ADL Dependency Score"),
    term = str_replace_all(term, "fdistress_inv", "Financial Distress"),
    term = str_replace_all(term, "owner", "Home Owner: Yes"),
    term = str_replace_all(term, "chronic", "Comorbidities"),
    term = factor(term, levels = rev(unique(term))),
    Period = factor(Period, levels = c("Pre-COVID-19 (2016-2019)", "During COVID-19 (2020-2021)"))
  )

# Generate comparative Forest Plot
p_covid_forest <- ggplot(coef_covid_df, aes(x = estimate, y = term, color = Period)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_point(position = position_dodge(width = 0.6), size = 3.5) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), position = position_dodge(width = 0.6), height = 0.3, linewidth = 0.8) +
  scale_color_manual(values = c("Pre-COVID-19 (2016-2019)" = "#2c3e50", "During COVID-19 (2020-2021)" = "#e74c3c")) +
  labs(
    x = "Effect Size in PPS Euros (€)", y = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(face = "bold", color = "black", size=11)
  )

ggsave(file.path(output_dir, "Appendix_Fig2_COVID_Robustness.png"), plot = p_covid_forest, width = 11, height = 7, dpi = 300, bg="white")

message(">>> COVID-19 Robustness Analysis successfully exported.")

# ==============================================================================
# 12.3 METHODOLOGICAL APPENDIX: UNOBSERVED CONFOUNDING (SENSEMAKR)
# ==============================================================================
message(">>> Running Sensemakr for Unobserved Confounding (Cinelli & Hazlett, 2020)...")

if(!require(sensemakr)) install.packages("sensemakr")
library(sensemakr)

model_total_lm <- lm(formula_cate_total, data = df_econ)

sense_clinical <- sensemakr(model = model_total_lm,
                            treatment = "cod_trajectoryOther",
                            benchmark_covariates = "eol_adl_score", 
                            kd = 1:3)

sink(file.path(output_dir, "Appendix_Sensemakr_Summary.txt"))
print(summary(sense_clinical))
sink()

sense_clinical$bounds$bound_label <- c("1x EoL ADL Score", "2x EoL ADL Score", "3x EoL ADL Score")

png(file.path(output_dir, "Appendix_Fig3_Unobserved_Confounding.png"), width = 2400, height = 1800, res = 300)
par(mar = c(5, 5, 4, 2) + 0.1)

# Expand contour limits to accommodate specific bounds
plot(sense_clinical, 
     type = "contour", 
     lim = 0.15, 
     xlab = "Partial R2 of Unobserved Confounder with Treatment",
     ylab = "Partial R2 of Unobserved Confounder with Outcome",
     cex.main = 1.1,
     cex.lab = 1.0,
     cex.axis = 0.9)

dev.off()

message(">>> Sensemakr analysis successfully exported in High Definition.")

message(">>> PIPELINE COMPLETE.")
