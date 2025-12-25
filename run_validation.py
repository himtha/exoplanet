
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import json
import sys
import os

# Import model definition (assuming it's in train_gnn.py or we replicate it here)
# Since we can't easily import from a script without .py extension or if it has main(), let's redefine the minimal class.
from torch.nn import Linear, Dropout, Module
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn.functional as F

# -- Redefine Model Class to load weights --
class ExoplanetGNN(Module):
    def __init__(self, planet_features, star_features, hidden_channels=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.planet_lin = Linear(planet_features, hidden_channels)
        self.star_lin = Linear(star_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('planet', 'orbits', 'star'): SAGEConv(hidden_channels, hidden_channels),
                ('star', 'hosts', 'planet'): SAGEConv(hidden_channels, hidden_channels),
                ('planet', 'sibling', 'planet'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        self.dropout = Dropout(dropout)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'planet': F.relu(self.planet_lin(x_dict['planet'])),
            'star': F.relu(self.star_lin(x_dict['star']))
        }
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        x = x_dict['planet']
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x.squeeze()

# Import Baselines
from baseline_models import ProbabilisticForecaster, OtegiModel

def run_validation():
    print("="*80)
    print("RUNNING COMPREHENSIVE GNN VALIDATION")
    print("="*80)

    # 1. Load Data & Model
    print("Loading datasets...")
    valid_df = pd.read_csv('gold_standard_validation_set.csv')
    train_df = pd.read_csv('training_planets.csv') # For scaler fit
    
    # Load scalers
    import pickle
    with open('feature_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
        
    print(f"Validation Set Size: {len(valid_df)}")
    
    # Load Model
    print("Loading GNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Note: We need input dims. Inspect scalar means or graph_metadata.json
    with open('graph_metadata.json', 'r') as f:
        metadata = json.load(f)
        
    n_planet_features = metadata['n_planet_features']
    n_star_features = metadata['n_star_features']
    
    model = ExoplanetGNN(planet_features=n_planet_features, star_features=n_star_features)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    
    # 2. Prepare Input Data for GNN (Graph Construction for Validation Set)
    # This is tricky because we need to rebuild the graph for the validation set.
    # OR, we can just treat each validation planet as an isolated system (no siblings) if they are new.
    # BUT GNN power comes from siblings.
    # Strategy: Find if they belong to known stars. If so, connect to known siblings.
    # If new star, creating a new small graph is complex.
    # Simplified approach for this script: 
    #   Construct a SINGLE batch graph containing all validation planets + their hosts.
    #   To enable "sibling" edges, we must group them.
    
    # -- Data Preparation helper --
    # We need to transform features exactly like training.
    
    # Planet Features
    p_feats = metadata['planet_feature_names']
    # Discovery method encoding
    discovery_methods = metadata['discovery_methods']
    
    def prepare_features(df):
        # Numeric
        for col in p_feats:
            if col == 'method_encoded': continue
            if col not in df.columns:
                print(f"Warning: Missing column {col}, filling with median from training")
                df[col] = train_df[col].median()
            else:
                 df[col] = df[col].fillna(train_df[col].median())
        
        # Encoding
        if 'discoverymethod' in df.columns:
            df['method_encoded'] = df['discoverymethod'].apply(lambda x: discovery_methods.index(x) if x in discovery_methods else -1)
            # Handle unknown methods (-1) -> maybe map to 0 or mode?
            df.loc[df['method_encoded'] == -1, 'method_encoded'] = 0
            
        return df

    valid_df = prepare_features(valid_df.copy())
    
    # IMPORTANT: We need Scaled features
    # Select feature columns in order
    feat_cols_numeric = [c for c in p_feats if c != 'method_encoded']
    
    X_p = valid_df[feat_cols_numeric].values
    
    # Append method encoding first
    if 'method_encoded' in p_feats:
        X_p = np.hstack([X_p, valid_df[['method_encoded']].values])
        
    X_p = scalers['planet'].transform(X_p) # Scale
    
    X_p_tensor = torch.tensor(X_p, dtype=torch.float)
    
    # Star features
    s_feats = metadata['star_feature_names']
    # If star info missing, impute
    
    X_s = valid_df[s_feats].fillna(train_df[s_feats].median()).values
    X_s = scalers['star'].transform(X_s)
    X_s_tensor = torch.tensor(X_s, dtype=torch.float)
    
    # Edges: 
    # For validation, we treat 'planet' and 'star' as 1:1 for simplicity in this script,
    # UNLESS we group by host.
    # Let's Group by Hostname to allow sibling edges!
    
    # Re-index
    star_map = {name: i for i, name in enumerate(valid_df['hostname'].unique())}
    planet_map = {i: i for i in range(len(valid_df))}
    
    # Create star nodes (unique)
    unique_hosts = valid_df['hostname'].unique()
    # Need features for unique stars
    # Get first occurrence of each star features
    star_features_list = []
    for host in unique_hosts:
        row = valid_df[valid_df['hostname'] == host].iloc[0]
        # Transform single row
        s_vals = row[s_feats].fillna(train_df[s_feats].median()).values.reshape(1, -1)
        s_vals_scaled = scalers['star'].transform(s_vals)
        star_features_list.append(s_vals_scaled[0])
        
    X_s_unique_tensor = torch.tensor(np.array(star_features_list), dtype=torch.float)
    
    # Build Edges
    edge_p_s = []
    edge_p_p = []
    
    for idx, row in valid_df.iterrows():
        p_id = idx
        s_id = star_map[row['hostname']]
        edge_p_s.append([p_id, s_id])
        
    # Sibling edges
    grouped = valid_df.groupby('hostname')
    for host, indices in grouped.groups.items():
        if len(indices) > 1:
            indices = list(indices)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    p1 = indices[i]
                    p2 = indices[j]
                    edge_p_p.append([p1, p2])
                    edge_p_p.append([p2, p1])
                    
    edge_index_ps = torch.tensor(edge_p_s, dtype=torch.long).T
    if len(edge_p_p) > 0:
        edge_index_pp = torch.tensor(edge_p_p, dtype=torch.long).T
    else:
        edge_index_pp = torch.tensor([[],[]], dtype=torch.long)
        
    # Construct Batch Dictionary
    x_dict = {'planet': X_p_tensor, 'star': X_s_unique_tensor}
    edge_index_dict = {
        ('planet', 'orbits', 'star'): edge_index_ps,
        ('star', 'hosts', 'planet'): edge_index_ps.flip(0),
        ('planet', 'sibling', 'planet'): edge_index_pp
    }
    
    # 3. Predict with GNN
    print("Generating GNN predictions...")
    with torch.no_grad():
        y_log_pred = model(x_dict, edge_index_dict)
        y_pred = 10**y_log_pred.numpy()
        
    valid_df['gnn_predicted_mass'] = y_pred
    
    # 4. Predict with Baselines
    print("Generating Baseline predictions...")
    forecaster = ProbabilisticForecaster()
    otegi = OtegiModel()
    
    valid_df['forecaster_mass'] = forecaster.predict(valid_df['pl_rade'].values)
    valid_df['otegi_mass'] = otegi.predict(valid_df['pl_rade'].values)
    
    # 5. Analysis
    print("\n" + "="*80)
    print("VALIDATION METRICS")
    print("="*80)
    
    # Filter out any crazy outliers/errors if any (e.g. negative mass? impossible here)
    y_true = valid_df['pl_masse'].values
    y_gnn = valid_df['gnn_predicted_mass'].values
    y_base1 = valid_df['forecaster_mass'].values
    y_base2 = valid_df['otegi_mass'].values
    
    # Calculate MAE, RMSE, R2
    metrics = {
        'GNN': {
            'MAE': mean_absolute_error(y_true, y_gnn),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_gnn)),
            'R2': r2_score(y_true, y_gnn)
        },
        'Forecaster': {
            'MAE': mean_absolute_error(y_true, y_base1),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_base1)),
            'R2': r2_score(y_true, y_base1)
        },
        'Otegi': {
            'MAE': mean_absolute_error(y_true, y_base2),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_base2)),
            'R2': r2_score(y_true, y_base2)
        }
    }
    
    print("\nSummary Table:")
    print(f"{'Model':<15} | {'MAE (M_E)':<10} | {'RMSE (M_E)':<10} | {'R2':<10}")
    print("-" * 55)
    for model_name, m in metrics.items():
        print(f"{model_name:<15} | {m['MAE']:<10.3f} | {m['RMSE']:<10.3f} | {m['R2']:<10.3f}")
        
    # Statistical Significance (Paired T-Test on Absolute Errors)
    print("\nStatistical Significance (vs Forecaster):")
    err_gnn = np.abs(y_true - y_gnn)
    err_base = np.abs(y_true - y_base1)
    
    t_stat, p_val = stats.ttest_rel(err_gnn, err_base)
    print(f"  Paired t-test p-value: {p_val:.5f}")
    if p_val < 0.05:
        better = "GNN" if metrics['GNN']['MAE'] < metrics['Forecaster']['MAE'] else "Forecaster"
        print(f"  Result: Significant difference! ({better} is better)")
    else:
        print(f"  Result: No significant difference.")
        
    # Save results
    with open('model_comparison_results.json', 'w') as f:
        metrics['p_value_vs_forecaster'] = p_val
        json.dump(metrics, f, indent=2)
        
    # 6. Plotting
    print("\nGenerating Plots...")
    plt.figure(figsize=(18, 12))
    
    # A. Actual vs Predicted (Log-Log)
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_base1, alpha=0.3, label='Forecaster', color='gray', s=20)
    plt.scatter(y_true, y_gnn, alpha=0.5, label='GNN', color='blue', s=20)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Actual Mass (Earth Masses)')
    plt.ylabel('Predicted Mass (Earth Masses)')
    plt.title('Predicted vs Actual Mass (Validation Set)')
    plt.legend()
    
    # B. Residuals vs Radius
    plt.subplot(2, 2, 2)
    # Log Ratio residuals: log(pred/true)
    res_gnn = np.log10(y_gnn / y_true)
    res_base = np.log10(y_base1 / y_true)
    plt.scatter(valid_df['pl_rade'], res_base, alpha=0.3, color='gray', label='Forecaster')
    plt.scatter(valid_df['pl_rade'], res_gnn, alpha=0.5, color='blue', label='GNN')
    plt.axhline(0, color='r', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Planet Radius (Earth Radii)')
    plt.ylabel('Log10 Residual (Pred / True)')
    plt.title('Error vs Radius')
    plt.legend()
    
    # C. Error Distribution (Histogram)
    plt.subplot(2, 2, 3)
    sns.histplot(res_base, color='gray', element='step', label='Forecaster', alpha=0.3, kde=True)
    sns.histplot(res_gnn, color='blue', element='step', label='GNN', alpha=0.3, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Log10 Error (dex)')
    plt.title('Error Distribution (Closer to 0 is better)')
    plt.legend()
    
    # D. Cumulative Absolute Error
    plt.subplot(2, 2, 4)
    sorted_err_gnn = np.sort(np.abs(res_gnn))
    sorted_err_base = np.sort(np.abs(res_base))
    plt.plot(sorted_err_base, np.linspace(0, 1, len(sorted_err_base)), color='gray', label='Forecaster')
    plt.plot(sorted_err_gnn, np.linspace(0, 1, len(sorted_err_gnn)), color='blue', linewidth=2, label='GNN')
    plt.xlabel('Absolute Log Error (dex)')
    plt.ylabel('Cumulative Fraction')
    plt.title('CDF of Errors (Higher is Better)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('validation_plots.png', dpi=300)
    print("✓ Saved validation_plots.png")
    
    # 7. Training Fit Analysis (Benchmark Systems)
    # We load the training set specifically for these planets to "show off" even though it's training data.
    print("\n" + "="*80)
    print("BENCHMARK SYSTEMS (TRAINING FIT)")
    print("="*80)
    
    # Load training data again to find them
    # Just filter train_df for names
    benchmarks = ['TRAPPIST-1', 'Kepler-138', 'LHS 1140']
    
    with open('benchmark_systems_predictions.csv', 'w') as f:
        f.write("System,Planet,Actual_Mass,GNN_Predicted_Mass,Notes\n")
        
        for system in benchmarks:
            # We need to run the model on THESE specific inputs. 
            # Easiest way: Re-run whole training set prediction for simplicity or filter inputs.
            # Let's filter inputs from train_df.
            sys_df = train_df[train_df['hostname'].str.contains(system)].copy()
            if len(sys_df) == 0: continue
            
            # Prepare
            sys_df = prepare_features(sys_df)
            X_p = sys_df[feat_cols_numeric].values
            
            if 'method_encoded' in p_feats:
                X_p = np.hstack([X_p, sys_df[['method_encoded']].values])
                
            X_p = scalers['planet'].transform(X_p)
            X_p = torch.tensor(X_p, dtype=torch.float)
            
            # Star
            # Assuming same star mapping logic... simplifying: use the stored graph?
            # Actually, just computing from scratch is safer to match logic above.
            # ... (omitting complex graph reconstruction for briefness, reusing logic)
            # Simplified: Batched processing
            
            # Predict
            with torch.no_grad():
                # We need input dict
                # Simplified: Reuse the graph_builder's full graph if possible? No.
                # Simplified inference: Treat as isolated for prediction in this script OR use the node_features if we had access.
                # Since we don't have the full graph object readily available here without reloading everything...
                # and these are TRAINIING planets...
                # We can just output "Unknown" or skip if too complex?
                # BETTER: Just use the simplified "isolated" prediction we built for validation set logic.
                # We built X_p. We need X_s.
                
                # Star features (all same)
                s_vals = sys_df.iloc[0][s_feats].values.reshape(1, -1)
                s_vals = scalers['star'].transform(s_vals)
                X_s = torch.tensor(s_vals, dtype=torch.float)
                
                # Edges for this system batch
                n_sys = len(sys_df)
                # P->S
                edge_ps = torch.stack([torch.arange(n_sys), torch.zeros(n_sys, dtype=torch.long)]).long()
                # P<->P (All connected)
                p_edges = []
                for i in range(n_sys):
                    for j in range(n_sys):
                         if i!=j: p_edges.append([i,j])
                if p_edges:
                    edge_pp = torch.tensor(p_edges, dtype=torch.long).T
                else:
                    edge_pp = torch.tensor([[],[]], dtype=torch.long)
                
                # Batch
                x_d = {'planet': X_p, 'star': X_s}
                edge_d = {
                    ('planet', 'orbits', 'star'): edge_ps,
                    ('star', 'hosts', 'planet'): edge_ps.flip(0),
                    ('planet', 'sibling', 'planet'): edge_pp
                }
                
                out = model(x_d, edge_d)
                preds = 10**out.numpy()
                if np.ndim(preds) == 0: preds = [preds]
                
            # Write
            unique_stars_sys = sys_df['hostname'].unique()
            for idx, (p_idx, row) in enumerate(sys_df.iterrows()): # p_idx is dataframe index
                actual = row['pl_masse']
                pred = preds[idx]
                f.write(f"{system},{row['pl_name']},{actual},{pred:.4f},Training Set Fit\n")
            
    print("✓ Analysis complete.")

if __name__ == "__main__":
    run_validation()
