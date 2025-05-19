# src/model_architectures/mlp.py
import torch
import torch.nn as nn

# Try to import from config, but make it usable standalone if config isn't directly accessible
MLP_DEFAULT_HIDDEN_UNITS = [128, 64]
MLP_DEFAULT_DROPOUT_RATE = 0.3
try:
    from ..config import MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE
except ImportError:
    # Fallback to defaults if config cannot be imported (e.g. when testing this file directly)
    print("Warning: config.py not found, using default MLP parameters for architecture.")
    MLP_HIDDEN_UNITS = MLP_DEFAULT_HIDDEN_UNITS
    MLP_DROPOUT_RATE = MLP_DEFAULT_DROPOUT_RATE


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_units: list = None, # Allow overriding config via args
                 output_dim: int = 1,
                 dropout_rate: float = None, # Allow overriding config via args
                 model_type: str = "regression"):
        super(SimpleMLP, self).__init__()
        self.model_type = model_type

        # Use passed arguments if provided, else fall back to config, then to local defaults
        _hidden_units = hidden_units if hidden_units is not None else MLP_HIDDEN_UNITS
        _dropout_rate = dropout_rate if dropout_rate is not None else MLP_DROPOUT_RATE

        layers = []
        current_dim = input_dim
        for units in _hidden_units:
            layers.append(nn.Linear(current_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(_dropout_rate))
            current_dim = units
        
        layers.append(nn.Linear(current_dim, output_dim))
        # For binary classification with BCEWithLogitsLoss, no final activation needed here
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    # Quick test
    print("Testing SimpleMLP architecture...")
    test_model_reg = SimpleMLP(input_dim=10, model_type="regression")
    test_model_cls = SimpleMLP(input_dim=10, model_type="classification")
    dummy_input = torch.randn(5, 10) # Batch of 5, 10 features
    print("Regression model output shape:", test_model_reg(dummy_input).shape)
    print("Classification model output shape:", test_model_cls(dummy_input).shape)