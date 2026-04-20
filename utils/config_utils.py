"""Utilities for handling configuration parsing and resolution."""

from typing import Any, Dict
import torch.nn as nn


# Mapping di stringhe a classi di attivazione
ACTIVATION_FUNCTIONS = {
    "nn.ReLU": nn.ReLU,
    "nn.Tanh": nn.Tanh,
    "nn.Sigmoid": nn.Sigmoid,
    "nn.LeakyReLU": nn.LeakyReLU,
    "nn.ELU": nn.ELU,
    "nn.GELU": nn.GELU,
}


def resolve_policy_kwargs(policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risolve i riferimenti alle classi di attivazione nelle policy_kwargs.
    
    Converte stringhe come 'nn.ReLU' in oggetti classe effettivi.
    
    Args:
        policy_kwargs: dizionario di kwargs di policy potenzialmente contenente stringhe
                      che riferiscono a funzioni di attivazione
    
    Returns:
        Dizionario con riferimenti di classe risolti
    """
    if policy_kwargs is None:
        return None
    
    resolved = {}
    for key, value in policy_kwargs.items():
        if key == "activation_fn" and isinstance(value, str):
            # Risolvi la stringa in una classe di attivazione
            if value in ACTIVATION_FUNCTIONS:
                resolved[key] = ACTIVATION_FUNCTIONS[value]
            else:
                raise ValueError(
                    f"Activation function '{value}' not recognized. "
                    f"Available options: {list(ACTIVATION_FUNCTIONS.keys())}"
                )
        elif isinstance(value, dict):
            # Ricorri per dizionari annidati (e.g., net_arch)
            resolved[key] = resolve_policy_kwargs(value)
        else:
            resolved[key] = value
    
    return resolved
