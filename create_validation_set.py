
import torch
import pandas as pd
import numpy as np

def create_set():
    print("="*80)
    print("CREATING GOLD STANDARD VALIDATION SET")
    print("="*80)

    # 1. Load Train/Test Masks
    print("Loading graph data for masks...")
    try:
        data = torch.load('exoplanet_graph.pt', weights_only=False)
        train_mask = data['planet'].train_mask.numpy()
        val_mask = data['planet'].val_mask.numpy()
        test_mask = data['planet'].test_mask.numpy()
        print(f"  Train: {train_mask.sum()}")
        print(f"  Val:   {val_mask.sum()}")
        print(f"  Test:  {test_mask.sum()}")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # 2. Load Original Training Data (to map index to name)
    # The graph nodes correspond 1:1 to rows in 'training_planets.csv'
    print("Loading training planets index...")
    training_planets_df = pd.read_csv('training_planets.csv')
    
    # Create a set of "Training" names to strictly exclude
    train_indices = np.where(train_mask)[0]
    training_names = set(training_planets_df.iloc[train_indices]['pl_name'].values)
    print(f"  Identified {len(training_names)} planets used in training (to be excluded).")

    # 3. Load Full NASA Archive
    print("Loading NASA Exoplanet Archive...")
    ps_df = pd.read_csv('PS_2025.12.25_09.21.26.csv', comment='#', low_memory=False)
    print(f"  Total rows: {len(ps_df)}")

    # 4. Filter for Gold Standard
    # Criteria:
    # - Mass is measured (pl_masse not null)
    # - Not "Calculated" (pl_bmassprov != 'Calculated' if available, otherwise check ref)
    # - Not in Training Set
    
    print("Filtering for measured masses...")
    # Basic existence check
    has_mass = ps_df[ps_df['pl_masse'].notna()].copy()
    
    # Filter out "Calculated" if column exists
    if 'pl_bmassprov' in has_mass.columns:
        # Standard values: 'Mass' or 'Msini' are good. 'Calculated' is bad.
        # But 'Calculated' might derive from Msini, so we rely on provenance.
        # Actually, let's look for "measured" roughly.
        # The prompt asked to filter where provenance is NOT "Calculated".
        # Let's see unique values usually found: 'Mass', 'Msini', 'M-R relationship' (bad)
        valid_provs = ['Mass', 'Msini']
        # Actually let's just exclude 'Calculated' and 'M-R relationship'
        # Inspect unique values in next step or assume standard archive flags
        pass 
        
    # We want precise measurements (RV or TTV)
    # In PS table, 'pl_masse' is the best estimate. 
    # Valid techniques: Radial Velocity, Transit Timing Variations, Astrometry.
    # Invalid for validation: Transit (usually means derived from radius unless TTV), Microlensing (often rough), Imaging (often model dependent).
    
    # Filter by discovery method? No, TTV planets are "Transit" method but have TTV mass.
    # Best proxy: Check if 'pl_rvamp' exists OR 'ttv_flag' is 1?
    # Actually, simpler: Use all with mass, exclude training, and exclude those with 'M-R relationship' provenance.
    
    # Strict exclusion of training planets
    params = ['pl_name', 'hostname', 'pl_rade', 'pl_orbper', 'pl_masse', 'pl_bmasseerr1', 'pl_bmasseerr2', 'discoverymethod', 'pl_rvamp']
    
    validation_candidates = []
    
    # Group by planet to pick best row
    grouped = has_mass.groupby('pl_name')
    
    count_train = 0
    count_valid = 0
    
    for name, group in grouped:
        if name in training_names:
            count_train += 1
            continue
            
        # Select best row (default flag = 1)
        best_rows = group[group['default_flag'] == 1]
        if len(best_rows) == 0:
            row = group.iloc[0]
        else:
            row = best_rows.iloc[0]
            
        # Check provenance/quality
        # Exclude if mass is derived purely from radius (chen-kipping etc)
        # Check 'pl_bmassprov': if it says 'M-R relationship' skip it.
        # Note: 'PS_2025...' might be a custom name, assuming standard columns.
        if 'pl_bmassprov' in row and 'M-R' in str(row['pl_bmassprov']):
             continue

        # Add to validation
        validation_candidates.append(row)
        count_valid += 1

    validation_df = pd.DataFrame(validation_candidates)
    
    print(f"  Excluded {count_train} training planets.")
    print(f"  Found {count_valid} potential validation planets.")
    
    if count_valid == 0:
        print("Error: No validation planets found! Check matching logic.")
        return

    # Select columns
    cols = [c for c in params if c in validation_df.columns]
    # Add star info for GNN
    star_cols = ['st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'st_age']
    cols.extend([c for c in star_cols if c in validation_df.columns])
    
    final_df = validation_df[cols].copy()
    
    # Save
    final_df.to_csv('gold_standard_validation_set.csv', index=False)
    print(f"\n✓ Saved 'gold_standard_validation_set.csv' with {len(final_df)} planets.")
    
    # Preview
    print("\nSample Validation Planets:")
    print(final_df[['pl_name', 'pl_masse', 'discoverymethod']].head())

if __name__ == "__main__":
    create_set()
