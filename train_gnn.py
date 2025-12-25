#!/usr/bin/env python3
"""
Train GNN for Exoplanet Mass Prediction
Uses heterogeneous graph structure with planet and star nodes.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Module
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import pickle
from datetime import datetime
from tqdm import tqdm

sns.set_style("whitegrid")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ExoplanetGNN(Module):
    """
    Heterogeneous Graph Neural Network for exoplanet mass prediction.
    Uses separate message passing for different edge types.
    """

    def __init__(self, planet_features, star_features, hidden_channels=64, num_layers=3, dropout=0.2):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Input projection layers
        self.planet_lin = Linear(planet_features, hidden_channels)
        self.star_lin = Linear(star_features, hidden_channels)

        # Heterogeneous graph convolutional layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('planet', 'orbits', 'star'): SAGEConv(hidden_channels, hidden_channels),
                ('star', 'hosts', 'planet'): SAGEConv(hidden_channels, hidden_channels),
                ('planet', 'sibling', 'planet'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)

        # Dropout
        self.dropout = Dropout(dropout)

        # Output layers (for planets only)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

    def forward(self, x_dict, edge_index_dict):
        # Project input features to hidden dimension
        x_dict = {
            'planet': F.relu(self.planet_lin(x_dict['planet'])),
            'star': F.relu(self.star_lin(x_dict['star']))
        }

        # Graph convolutional layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Prediction head (planets only)
        x = x_dict['planet']
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x.squeeze()


class Trainer:
    """Handles training, validation, and evaluation of the GNN model."""

    def __init__(self, model, data, lr=0.001, weight_decay=1e-5, device='cpu'):
        self.model = model.to(device)
        self.data = data
        self.device = device

        # Move data to device
        self.data = self._move_data_to_device(data, device)

        # Optimizer and scheduler
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=10, min_lr=1e-6)

        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def _move_data_to_device(self, data, device):
        """Move HeteroData to device."""
        for key, value in data.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        data[key][k] = v.to(device)
            elif isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        return data

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(self.data.x_dict, self.data.edge_index_dict)

        # Compute loss on training set
        train_mask = self.data['planet'].train_mask
        loss = F.mse_loss(out[train_mask], self.data['planet'].y[train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate model on given mask."""
        self.model.eval()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        loss = F.mse_loss(out[mask], self.data['planet'].y[mask])
        return loss.item()

    def train(self, num_epochs=200, patience=20, verbose=True):
        """Main training loop with early stopping."""
        print("="*80)
        print("TRAINING GNN FOR EXOPLANET MASS PREDICTION")
        print("="*80)
        print(f"\nModel: {self.model.__class__.__name__}")
        print(f"Hidden channels: {self.model.hidden_channels}")
        print(f"Num layers: {self.model.num_layers}")
        print(f"Dropout: {self.model.dropout_rate}")
        print(f"Device: {self.device}")
        print(f"\nTraining samples: {self.data['planet'].train_mask.sum().item():,}")
        print(f"Validation samples: {self.data['planet'].val_mask.sum().item():,}")
        print(f"Test samples: {self.data['planet'].test_mask.sum().item():,}")
        print(f"\nStarting training for {num_epochs} epochs (patience={patience})...")
        print("="*80)

        # Progress bar
        pbar = tqdm(range(num_epochs), desc='Training')

        for epoch in pbar:
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.evaluate(self.data['planet'].val_mask)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best': f'{self.best_val_loss:.4f}'
            })

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1

            # Verbose logging every 10 epochs
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"\nEpoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n\n⚠️  Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break

        # Load best model
        print(f"\n✓ Training complete!")
        print(f"Loading best model from epoch {self.best_epoch}")
        self.model.load_state_dict(torch.load('best_model.pt'))

        return self.history

    @torch.no_grad()
    def predict(self, mask=None):
        """Get predictions for a given mask."""
        self.model.eval()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)

        if mask is not None:
            return out[mask].cpu().numpy(), self.data['planet'].y[mask].cpu().numpy()
        else:
            return out.cpu().numpy(), self.data['planet'].y.cpu().numpy()


def load_and_prepare_data(graph_path='exoplanet_graph.pt'):
    """Load graph and apply log transform to target."""
    print("Loading graph data...")
    data = torch.load(graph_path, weights_only=False)

    print(f"Graph loaded successfully!")
    print(f"  Planet nodes: {len(data['planet'].x):,}")
    print(f"  Star nodes: {len(data['star'].x):,}")

    # Apply log transform to target (planet masses)
    print("\nApplying log10 transform to target masses...")
    masses = data['planet'].y
    print(f"  Original range: {masses.min().item():.2f} - {masses.max().item():.2f} M⊕")
    print(f"  Original mean: {masses.mean().item():.2f} M⊕")

    # Log transform: log10(mass)
    data['planet'].y_original = masses.clone()  # Keep original for later
    data['planet'].y = torch.log10(masses)

    print(f"  Log-transformed range: {data['planet'].y.min().item():.2f} - {data['planet'].y.max().item():.2f}")
    print(f"  Log-transformed mean: {data['planet'].y.mean().item():.2f}")

    return data


def calculate_baseline_metrics(data):
    """Calculate baseline metrics (predict mean for everything)."""
    train_mask = data['planet'].train_mask
    val_mask = data['planet'].val_mask
    test_mask = data['planet'].test_mask

    # Calculate mean on training set (in log space)
    train_mean = data['planet'].y[train_mask].mean().item()

    # Predict mean for all
    pred_mean = torch.full_like(data['planet'].y, train_mean)

    # Calculate metrics on test set
    y_test = data['planet'].y[test_mask].cpu().numpy()
    pred_test = pred_mean[test_mask].cpu().numpy()

    baseline_mse = mean_squared_error(y_test, pred_test)
    baseline_mae = mean_absolute_error(y_test, pred_test)
    baseline_r2 = r2_score(y_test, pred_test)

    # Convert back to original scale for interpretability
    y_test_orig = 10 ** y_test
    pred_test_orig = 10 ** pred_test

    baseline_mae_orig = mean_absolute_error(y_test_orig, pred_test_orig)
    baseline_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, pred_test_orig))

    print("\n" + "="*80)
    print("BASELINE METRICS (Predict Mean)")
    print("="*80)
    print(f"Predicting constant value: {train_mean:.2f} (log scale) = {10**train_mean:.2f} M⊕")
    print(f"\nLog-scale metrics:")
    print(f"  MSE: {baseline_mse:.4f}")
    print(f"  MAE: {baseline_mae:.4f}")
    print(f"  R²:  {baseline_r2:.4f}")
    print(f"\nOriginal-scale metrics:")
    print(f"  MAE:  {baseline_mae_orig:.2f} M⊕")
    print(f"  RMSE: {baseline_rmse_orig:.2f} M⊕")

    return {
        'mse': baseline_mse,
        'mae': baseline_mae,
        'r2': baseline_r2,
        'mae_original': baseline_mae_orig,
        'rmse_original': baseline_rmse_orig
    }


def evaluate_model(trainer, baseline_metrics):
    """Comprehensive model evaluation."""
    print("\n" + "="*80)
    print("MODEL EVALUATION ON TEST SET")
    print("="*80)

    # Get predictions
    pred_test, y_test = trainer.predict(trainer.data['planet'].test_mask)

    # Metrics in log space
    mse = mean_squared_error(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred_test)

    # Convert back to original scale
    pred_test_orig = 10 ** pred_test
    y_test_orig = 10 ** y_test

    mae_orig = mean_absolute_error(y_test_orig, pred_test_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, pred_test_orig))
    r2_orig = r2_score(y_test_orig, pred_test_orig)

    # Calculate improvement over baseline
    mae_improvement = ((baseline_metrics['mae'] - mae) / baseline_metrics['mae']) * 100
    r2_improvement = ((r2 - baseline_metrics['r2']) / abs(baseline_metrics['r2'])) * 100 if baseline_metrics['r2'] != 0 else 0

    print("\nLog-scale metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    print("\nOriginal-scale metrics:")
    print(f"  MAE:  {mae_orig:.2f} M⊕")
    print(f"  RMSE: {rmse_orig:.2f} M⊕")
    print(f"  R²:   {r2_orig:.4f}")

    print("\n" + "="*80)
    print("IMPROVEMENT OVER BASELINE")
    print("="*80)
    print(f"MAE improvement:  {mae_improvement:+.2f}%")
    print(f"R² improvement:   {r2_improvement:+.2f}%")

    metrics = {
        'test_mse': mse,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        'test_mae_original': mae_orig,
        'test_rmse_original': rmse_orig,
        'test_r2_original': r2_orig,
        'mae_improvement': mae_improvement,
        'r2_improvement': r2_improvement
    }

    return metrics, pred_test, y_test, pred_test_orig, y_test_orig


def create_evaluation_plots(trainer, pred_test, y_test, pred_test_orig, y_test_orig, baseline_metrics):
    """Create comprehensive evaluation plots."""
    print("\n" + "="*80)
    print("CREATING EVALUATION PLOTS")
    print("="*80)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Training history
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = trainer.history['epochs']
    ax1.plot(epochs, trainer.history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, trainer.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.axvline(x=trainer.best_epoch, color='red', linestyle='--', label=f'Best Epoch ({trainer.best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (log scale)')
    ax1.set_title('Training History', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Learning rate schedule
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, trainer.history['learning_rates'], color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)

    # 3. Loss improvement
    ax3 = fig.add_subplot(gs[0, 2])
    final_train_loss = trainer.history['train_loss'][-1]
    final_val_loss = trainer.history['val_loss'][-1]
    baseline_loss = baseline_metrics['mse']

    bars = ax3.bar(['Baseline\n(Mean)', 'Train\n(GNN)', 'Val\n(GNN)'],
                   [baseline_loss, final_train_loss, final_val_loss],
                   color=['gray', 'green', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('MSE Loss (log scale)')
    ax3.set_title('Loss Comparison', fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Predicted vs Actual (log scale)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_test, pred_test, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), pred_test.min())
    max_val = max(y_test.max(), pred_test.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax4.set_xlabel('Actual log₁₀(Mass) [M⊕]')
    ax4.set_ylabel('Predicted log₁₀(Mass) [M⊕]')
    ax4.set_title(f'Predicted vs Actual (Log Scale)\nR² = {r2_score(y_test, pred_test):.4f}',
                 fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Predicted vs Actual (original scale)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_test_orig, pred_test_orig, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    min_val = min(y_test_orig.min(), pred_test_orig.min())
    max_val = max(y_test_orig.max(), pred_test_orig.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax5.set_xlabel('Actual Mass [M⊕]')
    ax5.set_ylabel('Predicted Mass [M⊕]')
    ax5.set_title(f'Predicted vs Actual (Original Scale)\nR² = {r2_score(y_test_orig, pred_test_orig):.4f}',
                 fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Residuals (log scale)
    ax6 = fig.add_subplot(gs[1, 2])
    residuals = pred_test - y_test
    ax6.scatter(y_test, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Actual log₁₀(Mass) [M⊕]')
    ax6.set_ylabel('Residuals (Predicted - Actual)')
    ax6.set_title('Residual Plot (Log Scale)', fontweight='bold')
    ax6.grid(alpha=0.3)

    # 7. Residuals distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax7.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax7.set_xlabel('Residuals (Log Scale)')
    ax7.set_ylabel('Frequency')
    ax7.set_title(f'Residuals Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}',
                 fontweight='bold')
    ax7.legend()

    # 8. Error distribution (original scale)
    ax8 = fig.add_subplot(gs[2, 1])
    errors_orig = pred_test_orig - y_test_orig
    ax8.hist(errors_orig, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax8.set_xlabel('Error [M⊕]')
    ax8.set_ylabel('Frequency')
    ax8.set_title(f'Error Distribution (Original Scale)\nMAE: {np.abs(errors_orig).mean():.2f} M⊕',
                 fontweight='bold')
    ax8.legend()

    # 9. Performance by mass range
    ax9 = fig.add_subplot(gs[2, 2])
    # Bin by mass ranges
    mass_bins = [0, 1, 10, 100, 1000, 10000]
    bin_labels = ['<1', '1-10', '10-100', '100-1k', '>1k']
    bin_indices = np.digitize(y_test_orig, mass_bins)

    mae_by_bin = []
    for i in range(1, len(mass_bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            mae_by_bin.append(mean_absolute_error(y_test_orig[mask], pred_test_orig[mask]))
        else:
            mae_by_bin.append(0)

    bars = ax9.bar(bin_labels, mae_by_bin, color='purple', alpha=0.7, edgecolor='black', linewidth=2)
    ax9.set_xlabel('Mass Range [M⊕]')
    ax9.set_ylabel('MAE [M⊕]')
    ax9.set_title('Performance by Mass Range', fontweight='bold')
    ax9.set_yscale('log')
    for bar, mae_val in zip(bars, mae_by_bin):
        if mae_val > 0:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                    f'{mae_val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Exoplanet Mass Prediction - GNN Evaluation', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_results.png")


def save_results(trainer, metrics, baseline_metrics):
    """Save training results and predictions."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save training history
    history_dict = {
        'epochs': trainer.history['epochs'],
        'train_loss': trainer.history['train_loss'],
        'val_loss': trainer.history['val_loss'],
        'learning_rates': trainer.history['learning_rates'],
        'best_epoch': trainer.best_epoch,
        'best_val_loss': trainer.best_val_loss
    }

    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("✓ Saved: training_history.json")

    # Save all metrics
    all_metrics = {
        'baseline': baseline_metrics,
        'model': metrics,
        'improvement': {
            'mae': metrics['mae_improvement'],
            'r2': metrics['r2_improvement']
        },
        'timestamp': datetime.now().isoformat()
    }

    with open('evaluation_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("✓ Saved: evaluation_metrics.json")

    # Save test predictions
    pred_all, y_all = trainer.predict()
    pred_all_orig = 10 ** pred_all
    y_all_orig = 10 ** y_all

    test_mask = trainer.data['planet'].test_mask.cpu().numpy()

    predictions_dict = {
        'test_indices': np.where(test_mask)[0].tolist(),
        'predictions_log': pred_all[test_mask].tolist(),
        'actual_log': y_all[test_mask].tolist(),
        'predictions_original': pred_all_orig[test_mask].tolist(),
        'actual_original': y_all_orig[test_mask].tolist()
    }

    with open('test_predictions.json', 'w') as f:
        json.dump(predictions_dict, f, indent=2)
    print("✓ Saved: test_predictions.json")

    # Model already saved as best_model.pt
    print("✓ Model weights: best_model.pt")


def main():
    """Main training pipeline."""
    print("="*80)
    print("EXOPLANET MASS PREDICTION WITH GNN")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    data = load_and_prepare_data()

    # Calculate baseline
    baseline_metrics = calculate_baseline_metrics(data)

    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)

    model = ExoplanetGNN(
        planet_features=data['planet'].x.shape[1],
        star_features=data['star'].x.shape[1],
        hidden_channels=128,
        num_layers=3,
        dropout=0.2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        data=data,
        lr=0.001,
        weight_decay=1e-5,
        device=device
    )

    # Train model
    history = trainer.train(num_epochs=200, patience=20, verbose=True)

    # Evaluate model
    metrics, pred_test, y_test, pred_test_orig, y_test_orig = evaluate_model(trainer, baseline_metrics)

    # Create plots
    create_evaluation_plots(trainer, pred_test, y_test, pred_test_orig, y_test_orig, baseline_metrics)

    # Save results
    save_results(trainer, metrics, baseline_metrics)

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest model from epoch {trainer.best_epoch}")
    print(f"Test R² (log scale): {metrics['test_r2']:.4f}")
    print(f"Test R² (original): {metrics['test_r2_original']:.4f}")
    print(f"Test MAE: {metrics['test_mae_original']:.2f} M⊕")
    print(f"Improvement over baseline: {metrics['mae_improvement']:+.2f}%")
    print("\n" + "="*80)
    print("Output files:")
    print("  • best_model.pt - Trained model weights")
    print("  • training_history.json - Loss history")
    print("  • evaluation_metrics.json - All metrics")
    print("  • test_predictions.json - Test set predictions")
    print("  • training_results.png - Comprehensive plots")
    print("="*80)


if __name__ == "__main__":
    main()
