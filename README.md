# Exoplanet-GNN: Mass Prediction with Graph Neural Networks 🪐

A Graph Neural Network (GNN) model designed to predict exoplanet masses by leveraging the contextual relationships between planets, their host stars, and their siblings in multi-planet systems.

## 🚀 Key Results
I rigorously validated this model against a **Gold Standard held-out set** of 425 planets with measured masses (strictly excluding training data).

*   **Statistically Significant Improvement:** The GNN outperformed the industry-standard probabilistic forecaster (Chen & Kipping, 2017) with $p < 10^{-7}$.
*   **Physics-Aware:** The model successfully differentiates between dense "Super-Earths" and "Mini-Neptunes" where radius-only models fail.
*   **Conservative Predictions:** The model is robust for candidate validation but conservative regarding exotic outliers (e.g., "cotton candy" planets).

📄 **[Read the Full Validation Report](VALIDATION_REPORT.md)**

## 📂 Repository Structure
*   `train_gnn.py`: Main script to train the Heterogeneous GNN.
*   `build_graph_structure.py`: Constructs the graph from NASA Exoplanet Archive data.
*   `run_validation.py`: Runs the validation suite against baselines.
*   `baseline_models.py`: Implementations of Chen & Kipping (2017) and Otegi (2020) models.
*   `create_validation_set.py`: Generates the Gold Standard validation dataset.
*   `visualize_graph_stats.py`: Generates stats about the graph structure.

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/himtha/exoplanet.git
cd exoplanet

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Usage

### 1. Train the Model
```bash
python train_gnn.py
```
This will save the best model to `best_model.pt`.

### 2. Run Validation
```bash
python run_validation.py
```
This generates the `validation_plots.png` and `model_comparison_results.json`.

## 📊 Methodology
Traditional mass-radius relations: $M = f(R)$.
**My Approach:** $M = f(R, Star, Siblings)$.
I construct a heterogeneous graph where:
*   **Nodes:** Planets and Stars.
*   **Edges:** `(Planet, orbits, Star)`, `(Star, hosts, Planet)`, `(Planet, sibling, Planet)`.

This allows the network to learn "system-level" physics.

## 🤝 Contributing
Open source for scientific collaboration. Feel free to open issues or PRs.
