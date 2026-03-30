import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

def plot_fidelity_comparison(real_series, syn_series, title, xlabel, output_name, fill_color_syn='#27ae60'):
    plt.figure(figsize=(9, 6))
    max_threshold = max(np.percentile(real_series.dropna(), 95), np.percentile(syn_series.dropna(), 95))
    r_vis = real_series[real_series <= max_threshold]
    s_vis = syn_series[syn_series <= max_threshold]
    
    sns.kdeplot(r_vis, label='Empirical Cohort', fill=True, color='#2c3e50', alpha=0.5, linewidth=2)
    sns.kdeplot(s_vis, label='Synthetic Cohort', fill=True, color=fill_color_syn, alpha=0.4, linewidth=2, linestyle='--')
    
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"   [Plotting] Successfully exported: {output_name}")

def plot_shift_by_welfare(df, col_y0, col_y1, title, xlabel, output_name):
    df_melt = df.melt(id_vars=['welfare_group'], value_vars=[col_y0, col_y1], var_name='Cohort', value_name='Value')
    
    df_melt['Cohort'] = df_melt['Cohort'].replace({
        col_y0: 'Standard Care (Y0)', 
        col_y1: 'Palliative Care (Y1)'
    })
    
    # FIX: Calcola il 95° percentile per Welfare Group considerando ENTRAMBE le coorti
    # per garantire che l'asse X venga tagliato nello stesso punto per Y0 e Y1.
    def clip_95th_unified(group):
        threshold = np.percentile(group['Value'].dropna(), 95)
        return group[group['Value'] <= threshold]
    
    df_clean = df_melt.groupby('welfare_group', group_keys=False).apply(clip_95th_unified)
    
    g = sns.FacetGrid(df_clean, col="welfare_group", col_wrap=2, height=4, aspect=1.3, sharex=False, sharey=False)
    g.map_dataframe(sns.kdeplot, x="Value", hue="Cohort", fill=True, alpha=0.5, palette=['#2c3e50', '#e74c3c'], linewidth=2)
    g.set_titles(col_template="{col_name} Regime", fontweight='bold', size=13)
    g.set_axis_labels(xlabel, "Density")
    g.add_legend(title="Policy Scenario", bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle(title, fontsize=16, fontweight='bold')
    g.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plotting] Successfully exported stratified FacetGrid: {output_name}")

class FESTEvaluator:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, num_cols: list, cat_cols: list):
        self.common_cols = real_data.columns.intersection(synthetic_data.columns)
        self.real = real_data[self.common_cols].copy()
        self.syn = synthetic_data[self.common_cols].copy()
        self.num_cols = [c for c in num_cols if c in self.common_cols]
        self.cat_cols = [c for c in cat_cols if c in self.common_cols]
        for col in self.num_cols:
            self.real[col] = pd.to_numeric(self.real[col], errors='coerce').round(4)
            self.syn[col] = self.syn[col].fillna(self.syn[col].mean())
            self.real[col] = self.real[col].fillna(self.real[col].mean())
            self.syn[col] = self.syn[col].fillna(self.syn[col].mean())
        for col in self.cat_cols:
            self.real[col] = self.real[col].astype(str).str.strip().str.lower()
            self.syn[col] = self.syn[col].astype(str).str.strip().str.lower()
            mode_val = self.real[col].mode()[0]
            self.real[col] = self.real[col].replace('nan', mode_val)
            self.syn[col] = self.syn[col].replace('nan', mode_val)

    def evaluate_fidelity(self) -> dict:
        results = {}
        ks_scores = [1 - ks_2samp(self.real[col], self.syn[col])[0] for col in self.num_cols]
        results['KS_Score_Avg'] = np.mean(ks_scores)
        real_mean = self.real[self.num_cols].mean()
        syn_mean = self.syn[self.num_cols].mean()
        results['MAPE_Mean_Error'] = np.mean(np.abs((real_mean - syn_mean) / (real_mean + 1e-6)))
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        r_enc, s_enc = self.real.copy(), self.syn.copy()
        if self.cat_cols:
            full_cat = pd.concat([r_enc[self.cat_cols], s_enc[self.cat_cols]])
            enc.fit(full_cat)
            r_enc[self.cat_cols] = enc.transform(r_enc[self.cat_cols])
            s_enc[self.cat_cols] = enc.transform(s_enc[self.cat_cols])
        results['Frobenius_Corr_Diff'] = np.linalg.norm(r_enc.corr().fillna(0) - s_enc.corr().fillna(0))
        return results

    def evaluate_privacy(self) -> dict:
        results = {}
        
        # 1. Preparazione sicura di tutte le features (continue + categoriche)
        X_real, X_syn = self.real.copy(), self.syn.copy()
        
        if self.cat_cols:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Fittiamo su entrambi per avere lo stesso dizionario
            enc.fit(pd.concat([X_real[self.cat_cols], X_syn[self.cat_cols]]))
            X_real[self.cat_cols] = enc.transform(X_real[self.cat_cols])
            X_syn[self.cat_cols] = enc.transform(X_syn[self.cat_cols])
            
        # Riempiamo i NA continui
        X_real = X_real.fillna(0)
        X_syn = X_syn.fillna(0)

        # 2. DCR calcolato sull'intero spazio vettoriale (Privacy REALE)
        scaler = MinMaxScaler()
        r_scaled = scaler.fit_transform(X_real)
        s_scaled = scaler.transform(X_syn)
        
        # Sottocampionamento per performance se il dataset sintetico è enorme
        if len(s_scaled) > 3000:
            np.random.seed(42)
            s_scaled = s_scaled[np.random.choice(len(s_scaled), 3000, replace=False)]
            
        nbrs = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(r_scaled) # Distanza Euclidea
        distances, _ = nbrs.kneighbors(s_scaled)
        results['DCR_Mean'] = np.mean(distances)

        # 3. Adversarial Accuracy (Nessun Data Leakage)
        X_real['label'] = 0
        X_syn['label'] = 1
        combined = pd.concat([X_real, X_syn], axis=0)
        
        X = combined.drop('label', axis=1)
        y = combined['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        results['Adversarial_Accuracy'] = accuracy_score(y_test, clf.predict(X_test))
        
        return results

def compute_shift_metrics(df, col_y0, col_y1):
    stat, p_val = ks_2samp(df[col_y0].dropna(), df[col_y1].dropna())
    mean_y0 = df[col_y0].mean()
    mean_y1 = df[col_y1].mean()
    # Ora calcoliamo l'ATE e non più l'ATT
    return {'KS_Statistic_EffectSize': stat, 'P-Value': p_val, 'Mean_Y0': mean_y0, 'Mean_Y1': mean_y1, 'ATE': mean_y1 - mean_y0}

if __name__ == "__main__":
    print("====================================================================")
    print("   FEST FRAMEWORK & CAUSAL SHIFT ANALYSIS PIPELINE (ATE LEVEL)      ")
    print("====================================================================")
    
    categorical_cols = ['welfare_group', 'gender', 'isced', 'mstat', 'cod_trajectory', 'time_help_cat', 'palliative_access']
    continuous_cols = ['age', 'nchild', 'thinc2', 'hnetw', 'eol_adl_score', 'help_hrs_day_cont', 'oop_costs']

    print("\n[PART 1] ASSESSING GENERATIVE FIDELITY AND DISCLOSURE RISK")
    try:
        # Questo blocco rimane identico: confronta i dati veri di baseline con quelli sintetici di baseline
        df_real_true = pd.read_csv('share_prepared_for_tabddpm.csv')
        df_base_synth = pd.read_csv('tabddpm_synthetic_baseline.csv')
        
        evaluator = FESTEvaluator(df_real_true, df_base_synth, continuous_cols, categorical_cols)
        fest_results = {**evaluator.evaluate_fidelity(), **evaluator.evaluate_privacy()}
        
        print("\n--- FEST Framework Results ---")
        for k, v in fest_results.items(): print(f"    - {k}: {v:.4f}")
        pd.DataFrame([fest_results]).to_csv("FEST_TabDDPM_Evaluation.csv", index=False)
        
        plot_fidelity_comparison(
            df_real_true['oop_costs'], df_base_synth['oop_costs'],
            title='Generative Fidelity: Raw Out-of-Pocket Expenditure',
            xlabel='Raw Out-of-Pocket Costs (€)',
            output_name='FEST_Fidelity_OOP.png'
        )
    except Exception as e:
        print(f"[ERROR] FEST evaluation failed: {e}")

    print("\n[PART 2] EVALUATING CAUSAL SHIFT VIA SYNTHETIC COUNTERFACTUALS (FULL ATE)")
    try:
        df_cf = pd.read_csv('synthetic_counterfactuals_Q1.csv')
        
        print("   [Data] Loading Eurostat PPP dataset for economic harmonization...")
        df_eurostat = pd.read_csv('prc_ppp_ind$defaultview_linear_2_0.csv')
        df_eurostat = df_eurostat[
            (df_eurostat['na_item'] == 'PLI_EU27_2020') & 
            (df_eurostat['ppp_cat'] == 'A01') & 
            (df_eurostat['TIME_PERIOD'].isin([2017, 2018, 2019, 2020, 2021]))
        ]
        ppp_mean = df_eurostat.groupby('geo')['OBS_VALUE'].mean() / 100.0
        
        share_iso_map = {11:'AT', 12:'DE', 13:'SE', 14:'NL', 15:'ES', 16:'IT', 17:'FR', 18:'DK', 19:'EL', 
                         20:'CH', 23:'BE', 28:'CZ', 29:'PL', 31:'LU', 32:'HU', 33:'PT', 34:'SI', 35:'EE', 
                         47:'HR', 48:'LT', 51:'BG', 53:'CY', 55:'FI', 57:'LV', 59:'MT', 61:'RO', 63:'SK'}
        
        df_cf['iso_code'] = df_cf['country'].map(share_iso_map)
        df_cf['ppp_idx'] = df_cf['iso_code'].map(ppp_mean).fillna(1.0)
        
        df_cf['real_oop_costs_ppp'] = df_cf['real_oop_costs'] / df_cf['ppp_idx']
        df_cf['synth_oop_costs_ppp'] = df_cf['synth_oop_costs'] / df_cf['ppp_idx']
        
        print("   [Analysis] Computing ATE Shift Metrics (PPP Adjusted)...")
        oop_metrics = compute_shift_metrics(df_cf, 'real_oop_costs_ppp', 'synth_oop_costs_ppp')
        hours_metrics = compute_shift_metrics(df_cf, 'real_care_hours', 'synth_care_hours')
        
        print("\n--- Structural Welfare Loss (Financial Toxicity - PPS Euro) ---")
        print(f"    Expected Mean without Palliative Care (Y0): € {oop_metrics['Mean_Y0']:.2f}")
        print(f"    Expected Mean with Palliative Care (Y1):    € {oop_metrics['Mean_Y1']:.2f}")
        print(f"    Average Treatment Effect (ATE):             € {oop_metrics['ATE']:.2f}")
        
        print("\n--- Labor Substitution Effect (Caregiving Hours) ---")
        print(f"    Expected Mean without Palliative Care (Y0): {hours_metrics['Mean_Y0']:.1f} hrs")
        print(f"    Expected Mean with Palliative Care (Y1):    {hours_metrics['Mean_Y1']:.1f} hrs")
        print(f"    Average Treatment Effect (ATE):             {hours_metrics['ATE']:.1f} hrs")

        print("\n   [Plotting] Generating Welfare-Stratified Impact Plots...")
        plot_shift_by_welfare(
            df_cf, 'real_oop_costs_ppp', 'synth_oop_costs_ppp', 
            title='Policy Impact on Financial Toxicity by Welfare Regime', 
            xlabel='Out-of-Pocket Costs (PPS €)',
            output_name='Causal_Shift_OOP_Stratified.png'
        )
        
        plot_shift_by_welfare(
            df_cf, 'real_care_hours', 'synth_care_hours', 
            title='Policy Impact on Caregiving Burden by Welfare Regime', 
            xlabel='Total Annual Caregiving Hours',
            output_name='Causal_Shift_Hours_Stratified.png'
        )
        
        print("\n====================================================================")
        print("   PIPELINE COMPLETED. Results and Academic Plots are ready.")
        print("====================================================================")
        
    except Exception as e:
        print(f"[ERROR] Causal shift evaluation failed: {e}")