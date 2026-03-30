# ==============================================================================
# TITLE:        Synthetic Data Generation and Full ATE Causal Estimation
# METHOD:       TabDDPM + Full T-Learner Architecture (Y1 and Y0 prediction)
# ==============================================================================

import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(2026)

# ==============================================================================
# 1. EMPIRICAL DATA LOADING AND PREPROCESSING
# ==============================================================================
print(">>> 1. Loading empirical data (Full Cohort)...")
df = pd.read_csv('share_prepared_for_tabddpm.csv')

categorical_cols = ['welfare_group', 'gender', 'isced', 'mstat', 'cod_trajectory', 'time_help_cat', 'palliative_access']
continuous_cols = ['age', 'nchild', 'thinc2', 'hnetw', 'eol_adl_score', 'help_hrs_day_cont', 'oop_costs']

for col in categorical_cols:
    df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)
for col in continuous_cols:
    df[col] = df[col].astype(float)

learning_df = df.drop(columns=['country']).copy()

# A. Trasformazioni Non-Lineari (Gestione code grasse e valori negativi)
learning_df['oop_costs'] = np.log1p(learning_df['oop_costs'])
learning_df['thinc2'] = np.arcsinh(learning_df['thinc2'])
learning_df['hnetw'] = np.arcsinh(learning_df['hnetw'])

# B. Robust Scaling (Immunità agli outlier milionari)
scaler = RobustScaler()
learning_df[continuous_cols] = scaler.fit_transform(learning_df[continuous_cols])

loader = GenericDataLoader(learning_df, target_column="palliative_access", sensitive_columns=categorical_cols)
print(f"Dataset successfully loaded: {df.shape[0]} observations.")

# ==============================================================================
# 2. TAB-DDPM TRAINING
# ==============================================================================
print("\n>>> 2. Instantiating and training TabDDPM...")
tab_ddpm = Plugins().get("ddpm", n_iter=3000, batch_size=256)
tab_ddpm.fit(loader)
print(">>> Training completed successfully.")

# ==============================================================================
# 3. SYNTHETIC MANIFOLD GENERATION
# ==============================================================================
# Generiamo 30.000 pazienti per assicurarci di avere enormi campioni sia per i trattati che i non trattati
print("\n>>> 3. Generating the synthetic counterfactual manifold (N=30,000)...")

synth_data_loader = tab_ddpm.generate(count=30000)
df_synth_universe = synth_data_loader.dataframe()

# Ritrasformazioni inverse su TUTTO l'universo sintetico
df_synth_universe[continuous_cols] = scaler.inverse_transform(df_synth_universe[continuous_cols])
df_synth_universe['oop_costs'] = np.maximum(0, np.expm1(df_synth_universe['oop_costs']))
df_synth_universe['thinc2'] = np.sinh(df_synth_universe['thinc2'])
df_synth_universe['hnetw'] = np.sinh(df_synth_universe['hnetw'])

df_synth_universe['age'] = np.clip(np.round(df_synth_universe['age']), 0, 120)
df_synth_universe['nchild'] = np.maximum(0, np.round(df_synth_universe['nchild']))
df_synth_universe['help_hrs_day_cont'] = np.clip(df_synth_universe['help_hrs_day_cont'], 0, 24)
df_synth_universe['eol_adl_score'] = np.clip(np.round(df_synth_universe['eol_adl_score']), 0, 7)

# ==============================================================================
# 4. FULL T-LEARNER CAUSAL ESTIMATION (ATE)
# ==============================================================================
print("\n>>> 4. T-Learner: Training Model Y1 (Treated) and Model Y0 (Untreated)...")

features = ['welfare_group', 'age', 'gender', 'isced', 'mstat', 'nchild', 'thinc2', 'hnetw', 'eol_adl_score', 'cod_trajectory']

# Creiamo le dummies sull'intero universo per evitare disallineamenti di colonne
X_synth_univ = pd.get_dummies(df_synth_universe[features], drop_first=True).astype(int)

# MASCHERE DI TRATTAMENTO SINTETICHE
mask_synth_1 = df_synth_universe['palliative_access'].astype(float) == 1.0
mask_synth_0 = df_synth_universe['palliative_access'].astype(float) == 0.0

X_synth_1 = X_synth_univ[mask_synth_1]
X_synth_0 = X_synth_univ[mask_synth_0]

# --- ADDESTRAMENTO MODELLO 1 (Y1: Destino CON Cure Palliative) ---
rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1}

model_oop_1 = RandomForestRegressor(**rf_params).fit(X_synth_1, df_synth_universe.loc[mask_synth_1, 'oop_costs'])
model_cat_1 = RandomForestRegressor(**rf_params).fit(X_synth_1, df_synth_universe.loc[mask_synth_1, 'time_help_cat'].astype(float))
model_hrs_1 = RandomForestRegressor(**rf_params).fit(X_synth_1, df_synth_universe.loc[mask_synth_1, 'help_hrs_day_cont'])

# --- ADDESTRAMENTO MODELLO 0 (Y0: Destino SENZA Cure Palliative) ---
model_oop_0 = RandomForestRegressor(**rf_params).fit(X_synth_0, df_synth_universe.loc[mask_synth_0, 'oop_costs'])
model_cat_0 = RandomForestRegressor(**rf_params).fit(X_synth_0, df_synth_universe.loc[mask_synth_0, 'time_help_cat'].astype(float))
model_hrs_0 = RandomForestRegressor(**rf_params).fit(X_synth_0, df_synth_universe.loc[mask_synth_0, 'help_hrs_day_cont'])

# ==============================================================================
# 5. G-COMPUTATION SULLA POPOLAZIONE REALE (INFERENCE)
# ==============================================================================
print("\n>>> 5. Estimating Counterfactuals for the ENTIRE European Population...")

X_real = pd.get_dummies(df[features], drop_first=True).astype(int)
X_real = X_real.reindex(columns=X_synth_univ.columns, fill_value=0)

# Previsioni Universali per Y1 (Cosa succederebbe se TUTTI avessero cure palliative)
pred_oop_1 = np.maximum(0, model_oop_1.predict(X_real)) 
pred_cat_1 = np.clip(np.round(model_cat_1.predict(X_real)), 0, 5) 
pred_hrs_1 = np.clip(model_hrs_1.predict(X_real), 0, 24) 

# Previsioni Universali per Y0 (Cosa succederebbe se NESSUNO avesse cure palliative)
pred_oop_0 = np.maximum(0, model_oop_0.predict(X_real)) 
pred_cat_0 = np.clip(np.round(model_cat_0.predict(X_real)), 0, 5) 
pred_hrs_0 = np.clip(model_hrs_0.predict(X_real), 0, 24) 

# INCROCIO CAUSALE: Mixiamo dati reali e controfattuali in base al trattamento effettivo
is_treated = df['palliative_access'] == '1'

# Vettori Finali Y1 (Treated): Dati reali per chi era trattato, Gemelli Digitali per i non trattati
final_oop_1 = np.where(is_treated, df['oop_costs'], pred_oop_1)
final_cat_1 = np.where(is_treated, df['time_help_cat'].astype(float), pred_cat_1)
final_hrs_1 = np.where(is_treated, df['help_hrs_day_cont'], pred_hrs_1)

# Vettori Finali Y0 (Untreated): Gemelli Digitali per chi era trattato, Dati reali per i non trattati
final_oop_0 = np.where(is_treated, pred_oop_0, df['oop_costs'])
final_cat_0 = np.where(is_treated, pred_cat_0, df['time_help_cat'].astype(float))
final_hrs_0 = np.where(is_treated, pred_hrs_0, df['help_hrs_day_cont'])

# ==============================================================================
# 6. STOCHASTIC SMOOTHING E DATAFRAME EXPORT (R-READY)
# ==============================================================================
print("\n>>> 6. Applying stochastic smoothing and exporting ATE results...")

def stochastic_category_to_days(cat_array):
    days = []
    for cat in cat_array:
        # FIX: Assicurati che il check avvenga su un intero, dato che pred_cat_x restituisce float (es. 1.0)
        c = int(cat) 
        if c == 0:     days.append(0)
        elif c == 1:   days.append(np.random.randint(1, 30))     
        elif c == 2:   days.append(np.random.randint(30, 90))    
        elif c == 3:   days.append(np.random.randint(90, 180))   
        elif c == 4:   days.append(np.random.randint(180, 365))  
        else:          days.append(365)                          
    return np.array(days)

results_df = df[['country', 'palliative_access'] + features].copy()

# TRUCCO PER R: 'real' = Y0 (Assenza di cure), 'synth' = Y1 (Presenza di cure)
results_df['real_oop_costs'] = final_oop_0
results_df['real_care_hours'] = stochastic_category_to_days(final_cat_0) * final_hrs_0

results_df['synth_oop_costs'] = final_oop_1
results_df['synth_care_hours'] = stochastic_category_to_days(final_cat_1) * final_hrs_1

output_path = 'synthetic_counterfactuals_Q1.csv'
results_df.to_csv(output_path, index=False)
print(f">>> Counterfactuals successfully saved: {output_path}. Ready for R.")