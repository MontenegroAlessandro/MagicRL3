"""Utility per i grafici delle run RT-DeepRL.

  schema     i campi dell'indice: titoli, formattazione, ruolo nelle figure
  sources    quali progetti W&B esistono e con che convenzioni si leggono
  index      metadati delle run (una riga per run), in cache su /storage
  curves     curve di eval/diagnostiche e aggregazione sui seed
  select     filtri in stile riga di comando
  figure     dalla selezione alla figura: pipeline unica di CLI e selettore
  selection  selezioni salvate dal selettore (lettura, scrittura, migrazione)
  labels     nomi degli algoritmi e delle serie
  style      palette IBM colorblind e look "da paper"
  grid       disegno della griglia di pannelli
  webui      il selettore interattivo
"""
from . import (curves, figure, index, labels, paths, schema, select, selection,
               sources, style)  # noqa: F401

__all__ = ["curves", "figure", "index", "labels", "paths", "schema", "select",
           "selection", "sources", "style"]
