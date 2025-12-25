#!/usr/bin/env python3
"""
Build Graph Structure for Exoplanet GNN
Creates a heterogeneous graph with planet and star nodes for mass prediction.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData, Data
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json

sns.set_style("whitegrid")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class ExoplanetGraphBuilder:
    """Build graph structure for exoplanet GNN training."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.training_df = None
        self.star_id_map = {}
        self.planet_id_map = {}
        self.method_encoder = LabelEncoder()
        self.scalers = {}

    def load_and_filter_data(self):
        """Load data and filter for planets with mass measurements."""
        print("="*80)
        print("STEP 1: Loading and Filtering Data")
        print("="*80)

        print("\nLoading full dataset...")
        self.df = pd.read_csv(self.csv_path, comment='#', low_memory=False)
        print(f"Total planets in dataset: {len(self.df):,}")

        # Filter for planets with mass measurements
        print("\nFiltering for planets with mass measurements (pl_masse not null)...")
        self.training_df = self.df[self.df['pl_masse'].notna()].copy()
        print(f"Planets with mass data: {len(self.training_df):,}")

        # Check for required features
        required_cols = ['pl_name', 'hostname', 'pl_rade', 'pl_orbper',
                        'pl_orbsmax', 'st_teff', 'st_rad', 'st_mass']
        missing_data = []

        print("\nChecking data completeness for required features:")
        for col in required_cols:
            missing = self.training_df[col].isna().sum()
            pct = (missing / len(self.training_df)) * 100
            print(f"  {col:<20}: {missing:>5} missing ({pct:>5.2f}%)")
            if missing > 0:
                missing_data.append(col)

        # Handle missing values
        if missing_data:
            print(f"\n⚠️  Handling missing values in: {', '.join(missing_data)}")
            # Fill missing numerical values with median
            for col in missing_data:
                if col in self.training_df.columns and self.training_df[col].dtype in [np.float64, np.int64]:
                    median_val = self.training_df[col].median()
                    self.training_df[col].fillna(median_val, inplace=True)
                    print(f"  Filled {col} with median: {median_val:.2f}")

        print(f"\n✓ Final training dataset: {len(self.training_df):,} planets")
        return self.training_df

    def prepare_features(self):
        """Prepare node features for planets and stars."""
        print("\n" + "="*80)
        print("STEP 2: Preparing Node Features")
        print("="*80)

        # Planet features
        planet_feature_cols = {
            'pl_rade': 'Planet Radius (R⊕)',
            'pl_orbper': 'Orbital Period (days)',
            'pl_orbsmax': 'Semi-Major Axis (AU)',
            'pl_eqt': 'Equilibrium Temp (K)',
            'pl_insol': 'Insolation Flux',
            'pl_orbeccen': 'Orbital Eccentricity',
            'pl_orbincl': 'Orbital Inclination (deg)'
        }

        # Star features
        star_feature_cols = {
            'st_teff': 'Stellar Temp (K)',
            'st_rad': 'Stellar Radius (R☉)',
            'st_mass': 'Stellar Mass (M☉)',
            'st_met': 'Metallicity [Fe/H]',
            'st_logg': 'Surface Gravity (log g)',
            'st_age': 'Stellar Age (Gyr)'
        }

        print("\nPlanet Features:")
        available_planet_features = []
        for col, desc in planet_feature_cols.items():
            if col in self.training_df.columns:
                avail = self.training_df[col].notna().sum()
                pct = (avail / len(self.training_df)) * 100
                print(f"  {desc:<30}: {avail:>5}/{len(self.training_df)} ({pct:>5.1f}%)")
                if pct > 50:  # Use features with >50% completeness
                    available_planet_features.append(col)

        print("\nStellar Features:")
        available_star_features = []
        for col, desc in star_feature_cols.items():
            if col in self.training_df.columns:
                avail = self.training_df[col].notna().sum()
                pct = (avail / len(self.training_df)) * 100
                print(f"  {desc:<30}: {avail:>5}/{len(self.training_df)} ({pct:>5.1f}%)")
                if pct > 50:
                    available_star_features.append(col)

        # Add discovery method as categorical feature
        print("\nEncoding discovery method as categorical feature...")
        if 'discoverymethod' in self.training_df.columns:
            methods = self.training_df['discoverymethod'].fillna('Unknown')
            self.method_encoder.fit(methods)
            method_encoded = self.method_encoder.transform(methods)
            self.training_df['method_encoded'] = method_encoded
            print(f"  Discovered {len(self.method_encoder.classes_)} unique methods")
            available_planet_features.append('method_encoded')

        self.planet_features = available_planet_features
        self.star_features = available_star_features

        print(f"\n✓ Selected {len(self.planet_features)} planet features")
        print(f"✓ Selected {len(self.star_features)} star features")

        return available_planet_features, available_star_features

    def create_graph_structure(self):
        """Create the graph with planet and star nodes."""
        print("\n" + "="*80)
        print("STEP 3: Building Graph Structure")
        print("="*80)

        # Create unique star IDs
        unique_stars = self.training_df['hostname'].unique()
        self.star_id_map = {star: idx for idx, star in enumerate(unique_stars)}
        print(f"\nUnique host stars: {len(unique_stars):,}")

        # Create planet IDs
        self.training_df['planet_id'] = range(len(self.training_df))
        self.training_df['star_id'] = self.training_df['hostname'].map(self.star_id_map)

        # Group planets by host star to identify siblings
        planets_by_star = self.training_df.groupby('hostname')['planet_id'].apply(list).to_dict()

        # Count multi-planet systems
        single_planet_systems = sum(1 for planets in planets_by_star.values() if len(planets) == 1)
        multi_planet_systems = len(planets_by_star) - single_planet_systems

        print(f"Single-planet systems: {single_planet_systems:,}")
        print(f"Multi-planet systems: {multi_planet_systems:,}")

        # Create edges
        print("\nBuilding edge connections...")

        # Planet-to-Star edges
        planet_to_star_edges = []
        for idx, row in self.training_df.iterrows():
            planet_id = row['planet_id']
            star_id = row['star_id']
            planet_to_star_edges.append([planet_id, star_id])

        planet_to_star_edges = np.array(planet_to_star_edges).T
        print(f"  Planet → Star edges: {planet_to_star_edges.shape[1]:,}")

        # Planet-to-Planet edges (siblings in same system)
        planet_to_planet_edges = []
        for star_name, planet_ids in planets_by_star.items():
            if len(planet_ids) > 1:  # Multi-planet system
                # Create edges between all sibling planets (fully connected within system)
                for i, p1 in enumerate(planet_ids):
                    for p2 in planet_ids[i+1:]:
                        planet_to_planet_edges.append([p1, p2])
                        planet_to_planet_edges.append([p2, p1])  # Bidirectional

        if planet_to_planet_edges:
            planet_to_planet_edges = np.array(planet_to_planet_edges).T
            print(f"  Planet ↔ Planet edges: {planet_to_planet_edges.shape[1]:,}")
        else:
            planet_to_planet_edges = np.array([[], []], dtype=np.int64)
            print(f"  Planet ↔ Planet edges: 0")

        self.edge_index_planet_star = torch.tensor(planet_to_star_edges, dtype=torch.long)
        self.edge_index_planet_planet = torch.tensor(planet_to_planet_edges, dtype=torch.long)

        return planets_by_star

    def create_node_features(self):
        """Create feature matrices for planet and star nodes."""
        print("\n" + "="*80)
        print("STEP 4: Creating Node Feature Matrices")
        print("="*80)

        # Prepare planet features
        print("\nExtracting planet features...")
        planet_feature_data = self.training_df[self.planet_features].copy()

        # Fill missing values with median
        for col in self.planet_features:
            if planet_feature_data[col].isna().any():
                median_val = planet_feature_data[col].median()
                planet_feature_data[col].fillna(median_val, inplace=True)

        # Normalize features
        print("Normalizing planet features...")
        scaler_planet = StandardScaler()
        planet_features_normalized = scaler_planet.fit_transform(planet_feature_data)
        self.planet_node_features = torch.tensor(planet_features_normalized, dtype=torch.float)
        self.scalers['planet'] = scaler_planet

        print(f"  Planet feature matrix shape: {self.planet_node_features.shape}")

        # Prepare star features (aggregate from planets)
        print("\nExtracting star features...")
        star_feature_data = []
        star_names = sorted(self.star_id_map.keys(), key=lambda x: self.star_id_map[x])

        for star_name in star_names:
            star_planets = self.training_df[self.training_df['hostname'] == star_name]
            # Take first planet's stellar features (all planets in system share same star)
            star_features = star_planets.iloc[0][self.star_features].values
            star_feature_data.append(star_features)

        star_feature_data = np.array(star_feature_data)

        # Fill missing values
        for i in range(star_feature_data.shape[1]):
            col_data = star_feature_data[:, i].astype(float)
            nan_mask = pd.isna(col_data)
            if nan_mask.any():
                median_val = np.nanmedian(col_data[~nan_mask])
                col_data[nan_mask] = median_val
                star_feature_data[:, i] = col_data

        # Normalize star features
        print("Normalizing star features...")
        scaler_star = StandardScaler()
        star_features_normalized = scaler_star.fit_transform(star_feature_data)
        self.star_node_features = torch.tensor(star_features_normalized, dtype=torch.float)
        self.scalers['star'] = scaler_star

        print(f"  Star feature matrix shape: {self.star_node_features.shape}")

        # Target variable (planet mass)
        print("\nPreparing target variable (planet mass)...")
        self.planet_mass = torch.tensor(self.training_df['pl_masse'].values, dtype=torch.float)
        print(f"  Mass target shape: {self.planet_mass.shape}")
        print(f"  Mass range: {self.planet_mass.min():.2f} - {self.planet_mass.max():.2f} M⊕")

        return self.planet_node_features, self.star_node_features, self.planet_mass

    def create_train_val_test_split(self):
        """Split data into train/validation/test sets."""
        print("\n" + "="*80)
        print("STEP 5: Creating Train/Val/Test Split (70/15/15)")
        print("="*80)

        n_planets = len(self.training_df)
        indices = np.arange(n_planets)

        # First split: 70% train, 30% temp
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)

        # Second split: 15% val, 15% test
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Create masks
        train_mask = torch.zeros(n_planets, dtype=torch.bool)
        val_mask = torch.zeros(n_planets, dtype=torch.bool)
        test_mask = torch.zeros(n_planets, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        print(f"\nTrain set: {train_mask.sum().item():,} planets ({train_mask.sum().item()/n_planets*100:.1f}%)")
        print(f"Val set:   {val_mask.sum().item():,} planets ({val_mask.sum().item()/n_planets*100:.1f}%)")
        print(f"Test set:  {test_mask.sum().item():,} planets ({test_mask.sum().item()/n_planets*100:.1f}%)")

        return train_mask, val_mask, test_mask

    def compute_graph_statistics(self):
        """Compute and display graph statistics."""
        print("\n" + "="*80)
        print("STEP 6: Graph Statistics")
        print("="*80)

        n_planets = self.planet_node_features.shape[0]
        n_stars = self.star_node_features.shape[0]
        n_planet_star_edges = self.edge_index_planet_star.shape[1]
        n_planet_planet_edges = self.edge_index_planet_planet.shape[1]

        print(f"\nGraph Structure:")
        print(f"  Planet nodes:        {n_planets:>6,}")
        print(f"  Star nodes:          {n_stars:>6,}")
        print(f"  Total nodes:         {n_planets + n_stars:>6,}")
        print(f"\nEdge Statistics:")
        print(f"  Planet → Star edges: {n_planet_star_edges:>6,}")
        print(f"  Planet ↔ Planet edges: {n_planet_planet_edges:>6,}")
        print(f"  Total edges:         {n_planet_star_edges + n_planet_planet_edges:>6,}")

        # Calculate average degree
        # For planets: connections to star + connections to siblings
        planet_degrees = np.zeros(n_planets)
        planet_degrees += 1  # Each planet connects to its star

        if n_planet_planet_edges > 0:
            edge_array = self.edge_index_planet_planet.numpy()
            for i in range(n_planets):
                planet_degrees[i] += np.sum(edge_array[0] == i)

        avg_planet_degree = planet_degrees.mean()

        # For stars: number of planets orbiting
        star_degrees = np.zeros(n_stars)
        for star_id in range(n_stars):
            star_degrees[star_id] = np.sum(self.edge_index_planet_star[1].numpy() == star_id)

        avg_star_degree = star_degrees.mean()

        print(f"\nNode Degree Statistics:")
        print(f"  Avg planet degree:   {avg_planet_degree:.2f}")
        print(f"  Avg star degree:     {avg_star_degree:.2f}")
        print(f"  Max planet degree:   {planet_degrees.max():.0f}")
        print(f"  Max star degree:     {star_degrees.max():.0f}")

        # System statistics
        planets_by_star = self.training_df.groupby('hostname').size()
        isolated_planets = (planets_by_star == 1).sum()
        multi_planet_systems = (planets_by_star > 1).sum()

        print(f"\nSystem Statistics:")
        print(f"  Isolated planets (no siblings): {isolated_planets:>6,}")
        print(f"  Multi-planet systems:           {multi_planet_systems:>6,}")
        print(f"  Planets in multi-planet systems: {(planet_degrees > 1).sum():>6,}")

        stats = {
            'n_planets': n_planets,
            'n_stars': n_stars,
            'n_planet_star_edges': n_planet_star_edges,
            'n_planet_planet_edges': n_planet_planet_edges,
            'avg_planet_degree': avg_planet_degree,
            'avg_star_degree': avg_star_degree,
            'isolated_planets': isolated_planets,
            'multi_planet_systems': multi_planet_systems
        }

        return stats

    def create_pytorch_geometric_data(self):
        """Create PyTorch Geometric HeteroData object."""
        print("\n" + "="*80)
        print("STEP 7: Creating PyTorch Geometric HeteroData Object")
        print("="*80)

        data = HeteroData()

        # Add planet nodes
        data['planet'].x = self.planet_node_features
        data['planet'].y = self.planet_mass
        data['planet'].train_mask = self.train_mask
        data['planet'].val_mask = self.val_mask
        data['planet'].test_mask = self.test_mask
        data['planet'].node_id = torch.arange(len(self.planet_node_features))

        # Add star nodes
        data['star'].x = self.star_node_features
        data['star'].node_id = torch.arange(len(self.star_node_features))

        # Add edges
        data['planet', 'orbits', 'star'].edge_index = self.edge_index_planet_star
        data['star', 'hosts', 'planet'].edge_index = self.edge_index_planet_star.flip(0)

        if self.edge_index_planet_planet.shape[1] > 0:
            data['planet', 'sibling', 'planet'].edge_index = self.edge_index_planet_planet

        print(f"\n✓ Created HeteroData object:")
        print(data)

        self.hetero_data = data
        return data

    def save_graph_data(self, output_dir='.'):
        """Save processed graph data."""
        print("\n" + "="*80)
        print("STEP 8: Saving Processed Graph Data")
        print("="*80)

        # Save HeteroData
        torch.save(self.hetero_data, f'{output_dir}/exoplanet_graph.pt')
        print(f"✓ Saved graph data: {output_dir}/exoplanet_graph.pt")

        # Save metadata
        metadata = {
            'n_planet_features': len(self.planet_features),
            'n_star_features': len(self.star_features),
            'planet_feature_names': self.planet_features,
            'star_feature_names': self.star_features,
            'n_planets': len(self.training_df),
            'n_stars': len(self.star_id_map),
            'discovery_methods': self.method_encoder.classes_.tolist(),
            'star_id_map': self.star_id_map,
        }

        with open(f'{output_dir}/graph_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {output_dir}/graph_metadata.json")

        # Save scalers
        with open(f'{output_dir}/feature_scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"✓ Saved feature scalers: {output_dir}/feature_scalers.pkl")

        # Save training dataframe
        self.training_df.to_csv(f'{output_dir}/training_planets.csv', index=False)
        print(f"✓ Saved training data: {output_dir}/training_planets.csv")

        print(f"\n✓ All graph data saved successfully!")


def visualize_example_system(graph_builder, system_name='Kepler-90'):
    """Visualize an example multi-planet system."""
    print("\n" + "="*80)
    print(f"VISUALIZATION: {system_name} System Graph")
    print("="*80)

    # Get planets in this system
    system_planets = graph_builder.training_df[
        graph_builder.training_df['hostname'] == system_name
    ]

    if len(system_planets) == 0:
        # Try TRAPPIST-1 as backup
        system_name = 'TRAPPIST-1'
        system_planets = graph_builder.training_df[
            graph_builder.training_df['hostname'] == system_name
        ]

    if len(system_planets) == 0:
        # Find any multi-planet system
        multi_systems = graph_builder.training_df.groupby('hostname').size()
        multi_systems = multi_systems[multi_systems > 1].sort_values(ascending=False)
        if len(multi_systems) > 0:
            system_name = multi_systems.index[0]
            system_planets = graph_builder.training_df[
                graph_builder.training_df['hostname'] == system_name
            ]

    if len(system_planets) == 0:
        print("⚠️  No multi-planet systems found in training data")
        return

    print(f"\nSystem: {system_name}")
    print(f"Number of planets: {len(system_planets)}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Graph structure
    ax1 = axes[0]
    G = nx.Graph()

    # Add star node
    star_node = f"{system_name}\n(Star)"
    G.add_node(star_node, node_type='star')

    # Add planet nodes and edges
    planet_nodes = []
    for idx, planet in system_planets.iterrows():
        planet_node = f"{planet['pl_name']}\n({planet['pl_rade']:.2f} R⊕)"
        planet_nodes.append(planet_node)
        G.add_node(planet_node, node_type='planet')
        G.add_edge(star_node, planet_node)  # Planet-star edge

    # Add planet-planet edges (siblings)
    for i, p1 in enumerate(planet_nodes):
        for p2 in planet_nodes[i+1:]:
            G.add_edge(p1, p2)  # Sibling edge

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    star_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'star']
    planet_nodes_graph = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'planet']

    nx.draw_networkx_nodes(G, pos, nodelist=star_nodes,
                          node_color='gold', node_size=1500,
                          node_shape='*', ax=ax1, label='Star')
    nx.draw_networkx_nodes(G, pos, nodelist=planet_nodes_graph,
                          node_color='lightblue', node_size=800,
                          node_shape='o', ax=ax1, label='Planets')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray', ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)

    ax1.set_title(f'{system_name} Graph Structure', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axis('off')

    # Right plot: Feature information
    ax2 = axes[1]
    ax2.axis('off')

    # Create feature table
    feature_info = f"{'='*60}\n"
    feature_info += f"System: {system_name}\n"
    feature_info += f"{'='*60}\n\n"

    feature_info += f"STAR NODE FEATURES:\n"
    feature_info += f"{'-'*60}\n"
    star_data = system_planets.iloc[0]
    for feat in graph_builder.star_features:
        val = star_data[feat]
        if pd.notna(val):
            feature_info += f"  {feat:<20}: {val:.2f}\n"

    feature_info += f"\n{'='*60}\n"
    feature_info += f"PLANET NODE FEATURES (example: {system_planets.iloc[0]['pl_name']}):\n"
    feature_info += f"{'-'*60}\n"
    for feat in graph_builder.planet_features:
        if feat != 'method_encoded':
            val = system_planets.iloc[0][feat]
            if pd.notna(val):
                feature_info += f"  {feat:<20}: {val:.2f}\n"

    feature_info += f"\n{'='*60}\n"
    feature_info += f"TARGET VARIABLE:\n"
    feature_info += f"{'-'*60}\n"
    feature_info += f"  Planet Mass (pl_masse)\n"
    for idx, planet in system_planets.iterrows():
        feature_info += f"    {planet['pl_name']:<25}: {planet['pl_masse']:>8.2f} M⊕\n"

    feature_info += f"\n{'='*60}\n"
    feature_info += f"EDGE TYPES:\n"
    feature_info += f"{'-'*60}\n"
    feature_info += f"  Planet → Star (orbits):   {len(system_planets)} edges\n"
    feature_info += f"  Star → Planet (hosts):    {len(system_planets)} edges\n"
    n_sibling_edges = len(system_planets) * (len(system_planets) - 1)
    feature_info += f"  Planet ↔ Planet (sibling): {n_sibling_edges} edges\n"

    ax2.text(0.05, 0.95, feature_info, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('example_system_graph.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: example_system_graph.png")

    # Print system details
    print(f"\n{system_name} System Details:")
    print("-" * 80)
    for idx, planet in system_planets.iterrows():
        print(f"  {planet['pl_name']:<30} | Mass: {planet['pl_masse']:>8.2f} M⊕ | "
              f"Radius: {planet['pl_rade']:>6.2f} R⊕ | Period: {planet['pl_orbper']:>10.2f} days")


def main():
    """Main execution function."""
    print("="*80)
    print("EXOPLANET GRAPH STRUCTURE BUILDER")
    print("For GNN-based Planet Mass Prediction")
    print("="*80)

    # Initialize builder
    builder = ExoplanetGraphBuilder('PS_2025.12.25_09.21.26.csv')

    # Build graph step by step
    builder.load_and_filter_data()
    builder.prepare_features()
    builder.create_graph_structure()
    builder.create_node_features()
    builder.create_train_val_test_split()
    stats = builder.compute_graph_statistics()
    builder.create_pytorch_geometric_data()
    builder.save_graph_data()

    # Visualize example system
    visualize_example_system(builder)

    # Final summary
    print("\n" + "="*80)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print(f"  ✓ Created graph with {stats['n_planets']:,} planets and {stats['n_stars']:,} stars")
    print(f"  ✓ Generated {stats['n_planet_star_edges']:,} planet-star edges")
    print(f"  ✓ Generated {stats['n_planet_planet_edges']:,} planet-planet edges")
    print(f"  ✓ Split into 70/15/15 train/val/test sets")
    print(f"\n📁 Output Files:")
    print(f"  • exoplanet_graph.pt - PyTorch Geometric graph")
    print(f"  • graph_metadata.json - Feature names and mappings")
    print(f"  • feature_scalers.pkl - Feature normalization scalers")
    print(f"  • training_planets.csv - Full training dataset")
    print(f"  • example_system_graph.png - Visualization")
    print(f"\n🚀 Ready for GNN training!")
    print(f"\nTo load the graph for training:")
    print(f"  >>> import torch")
    print(f"  >>> data = torch.load('exoplanet_graph.pt')")
    print(f"  >>> print(data)")


if __name__ == "__main__":
    main()
