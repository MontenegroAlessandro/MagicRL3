"""Selettore interattivo: `api` (logica, senza HTTP) + `server` (trasporto)."""
from .server import serve

__all__ = ["serve"]
